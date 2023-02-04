package naive

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
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

// Probabilities runs the provided string through the model and returns
// the potential probability for each classification
func (c *Classifier) Probabilities(stringToClassify string) (map[string]float64, string) {
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	var features []string
	for feature := range c.Tokenizer.Tokenize(AsReader(stringToClassify)) {
		features = append(features, feature)
	}

	totalCount := float64(c.count())
	categories := c.categories()
	numberOfGroups := 10
	groupSize := int(math.Ceil(float64(len(categories)) / float64(numberOfGroups)))

	var lock sync.Mutex
	var wg sync.WaitGroup
	wg.Add(numberOfGroups)

	for i := 0; i < numberOfGroups; i++ {
		go probabilityGrouped(c, categories, features, totalCount, probabilities, &wg, i, groupSize, lock)
	}

	fmt.Println("Calculating probabilities...")
	wg.Wait()

	keys := make([]string, 0, len(probabilities))
	for category := range probabilities {
		keys = append(keys, category)
	}

	sort.SliceStable(keys, func(i, j int) bool {
		return probabilities[keys[i]] > probabilities[keys[j]]
	})

	topCategory := ""

	if len(keys) > 0 {
		topCategory = keys[0]
	}

	return probabilities, topCategory
}

func probabilityGrouped(c *Classifier, categories []string, features []string, totalCount float64, probabilities map[string]float64, wg *sync.WaitGroup, offset int, groupSize int, lock sync.Mutex) {
	defer wg.Done()
	probabilitiesForThisGroup := map[string]float64{}
	for i := offset; i < offset+groupSize; i++ {
		if i < len(categories) {
			probabilitiesForThisGroup[categories[i]] = c.probability(categories, features, totalCount, categories[i])
		}
	}

	lock.Lock()
	for key, value := range probabilitiesForThisGroup {
		probabilities[key] = value
	}
	lock.Unlock()
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

func (c *Classifier) weightedProbability(categories []string, feature string, category string) float64 {
	return c.variableWeightedProbability(categories, feature, category, 1.0, 0.5)
}

func (c *Classifier) variableWeightedProbability(categories []string, feature string, category string, weight float64, assumedProb float64) float64 {
	sum := 0.0
	probability := c.featureProbability(feature, category)
	for _, category := range categories {
		sum += c.featureCount(feature, category)
	}
	return ((weight * assumedProb) + (sum * probability)) / (weight + sum)
}

func (c *Classifier) probability(categories []string, features []string, totalCount float64, category string) float64 {
	categoryProbability := c.categoryCount(category) / totalCount
	docProbability := c.docProbability(categories, features, category)
	return docProbability * categoryProbability
}

func (c *Classifier) docProbability(categories []string, features []string, category string) float64 {
	probability := 1.0
	for _, feature := range features {
		probability *= c.weightedProbability(categories, feature, category)
	}
	return probability
}

func AsReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
