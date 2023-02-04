package naive

import (
	"bytes"
	"errors"
	"io"
	"sync"

	"github.com/carautenbach/classifier"
)

// ErrNotClassified indicates that a document could not be classified
var ErrNotClassified = errors.New("unable to classify document")

// Option provides a functional setting for the Classifier
type Option func(c *Classifier) error

// Classifier implements a naive bayes classifier
type Classifier struct {
	Feat2cat  map[string]map[string]int
	CatCount  map[string]int
	Tokenizer classifier.Tokenizer
	mu        sync.RWMutex
}

// New initializes a new naive Classifier using the standard tokenizer
func New(opts ...Option) *Classifier {
	c := &Classifier{
		Feat2cat:  make(map[string]map[string]int),
		CatCount:  make(map[string]int),
		Tokenizer: classifier.NewTokenizer(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Tokenizer overrides the classifier's default Tokenizer
func Tokenizer(t classifier.Tokenizer) Option {
	return func(c *Classifier) error {
		c.Tokenizer = t
		return nil
	}
}

// Train provides supervisory training to the classifier
func (c *Classifier) Train(r io.Reader, category string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for feature := range c.Tokenizer.Tokenize(r) {
		c.addFeature(feature, category)
	}

	c.addCategory(category)
	return nil
}

// TrainString provides supervisory training to the classifier
func (c *Classifier) TrainString(doc string, category string) error {
	return c.Train(AsReader(doc), category)
}

// ClassifyString attempts to classify a document. If the document cannot be classified
// (e.g. because the classifier has not been trained), an error is returned.
func (c *Classifier) ClassifyString(stringToClassify string) (string, error) {
	max := 0.0
	var err error
	classification := ""
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	var features []string
	for feature := range c.Tokenizer.Tokenize(AsReader(stringToClassify)) {
		features = append(features, feature)
	}

	for _, category := range c.categories() {
		probabilities[category] = c.probability(features, category)
		if probabilities[category] > max {
			max = probabilities[category]
			classification = category
		}
	}

	if classification == "" {
		return "", ErrNotClassified
	}
	return classification, err
}

// Probabilities runs the provided string through the model and returns
// the potential probability for each classification
func (c *Classifier) Probabilities(stringToClassify string) (map[string]float64, string) {
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	best := 0.0
	cat := ``

	var features []string
	for feature := range c.Tokenizer.Tokenize(AsReader(stringToClassify)) {
		features = append(features, feature)
	}

	for _, category := range c.categories() {
		prob := c.probability(features, category)
		if prob > 0 {
			probabilities[category] = prob
		}
		if prob > best {
			best = prob
			cat = category
		}
	}

	return probabilities, cat
}

func (c *Classifier) addFeature(feature string, category string) {
	if _, ok := c.Feat2cat[feature]; !ok {
		c.Feat2cat[feature] = make(map[string]int)
	}
	c.Feat2cat[feature][category]++
}

func (c *Classifier) featureCount(feature string, category string) float64 {
	if _, ok := c.Feat2cat[feature]; ok {
		return float64(c.Feat2cat[feature][category])
	}
	return 0.0
}

func (c *Classifier) addCategory(category string) {
	c.CatCount[category]++
}

func (c *Classifier) categoryCount(category string) float64 {
	if _, ok := c.CatCount[category]; ok {
		return float64(c.CatCount[category])
	}
	return 0.0
}

func (c *Classifier) count() int {
	sum := 0
	for _, value := range c.CatCount {
		sum += value
	}
	return sum
}

func (c *Classifier) categories() []string {
	var keys []string
	for k := range c.CatCount {
		keys = append(keys, k)
	}
	return keys
}

func (c *Classifier) featureProbability(feature string, category string) float64 {
	if c.categoryCount(category) == 0 {
		return 0.0
	}
	return c.featureCount(feature, category) / c.categoryCount(category)
}

func (c *Classifier) weightedProbability(feature string, category string) float64 {
	return c.variableWeightedProbability(feature, category, 1.0, 0.5)
}

func (c *Classifier) variableWeightedProbability(feature string, category string, weight float64, assumedProb float64) float64 {
	sum := 0.0
	probability := c.featureProbability(feature, category)
	for _, category := range c.categories() {
		sum += c.featureCount(feature, category)
	}
	return ((weight * assumedProb) + (sum * probability)) / (weight + sum)
}

func (c *Classifier) probability(features []string, category string) float64 {
	categoryProbability := c.categoryCount(category) / float64(c.count())
	docProbability := c.docProbability(features, category)
	return docProbability * categoryProbability
}

func (c *Classifier) docProbability(features []string, category string) float64 {
	probability := 1.0
	for _, feature := range features {
		probability *= c.weightedProbability(feature, category)
	}
	return probability
}

func AsReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
