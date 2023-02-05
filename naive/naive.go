package naive

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/carautenbach/classifier"
)

// Classifier implements a naive bayes classifier
type Classifier struct {
	Feat2cat  map[string]map[string]int
	CatCount  map[string]int
	Tokenizer classifier.Tokenizer
	mu        sync.RWMutex
}

// New initializes a new naive Classifier using the standard tokenizer
func New() *Classifier {
	c := &Classifier{
		Feat2cat:  make(map[string]map[string]int),
		CatCount:  make(map[string]int),
		Tokenizer: classifier.NewTokenizer(),
	}
	return c
}

// Train provides supervisory training to the classifier
func (c *Classifier) Train(r io.Reader, category string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for word := range c.Tokenizer.Tokenize(r) {
		c.addWord(word, category)
	}

	c.CatCount[category]++
	return nil
}

// TrainString provides supervisory training to the classifier
func (c *Classifier) TrainString(title string, category string) error {
	return c.Train(AsReader(title), category)
}

// Probabilities runs the provided string through the model and returns
// the potential probabilityForCategory for each classification
func (c *Classifier) Probabilities(stringToClassify string) (map[string]float64, string) {
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	var features []string
	for feature := range c.Tokenizer.Tokenize(AsReader(stringToClassify)) {
		features = append(features, feature)
	}

	categories := c.getAllCategories()
	numberOfGroups := 1
	groupSize := int(math.Ceil(float64(len(categories)) / float64(numberOfGroups)))

	var lock sync.Mutex
	var wg sync.WaitGroup
	wg.Add(numberOfGroups)

	for i := 0; i < numberOfGroups; i++ {
		go probabilityGrouped(c, categories, features, probabilities, &wg, i, groupSize, lock)
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

func probabilityGrouped(c *Classifier, categories []string, words []string, probabilities map[string]float64, wg *sync.WaitGroup, offset int, groupSize int, lock sync.Mutex) {
	defer wg.Done()
	probabilitiesForThisGroup := map[string]float64{}
	for i := offset; i < offset+groupSize; i++ {
		if i < len(categories) {
			probability := c.probabilityForCategory(categories, words, categories[i])
			if probability > 0 {
				probabilitiesForThisGroup[categories[i]] = probability
			}
		}
	}

	lock.Lock()
	for key, value := range probabilitiesForThisGroup {
		probabilities[key] = value
	}
	lock.Unlock()
}

func (c *Classifier) addWord(word string, category string) {
	if _, ok := c.Feat2cat[word]; !ok {
		c.Feat2cat[word] = make(map[string]int)
	}
	c.Feat2cat[word][category]++
}

func (c *Classifier) countOfWordInCategory(word string, category string) float64 {
	if _, ok := c.Feat2cat[word]; ok {
		return float64(c.Feat2cat[word][category])
	}
	return 0.0
}

func (c *Classifier) totalCountInCategory(category string) float64 {
	if _, ok := c.CatCount[category]; ok {
		return float64(c.CatCount[category])
	}
	return 0.0
}

func (c *Classifier) countOfAllResults() int {
	sum := 0
	for _, value := range c.CatCount {
		sum += value
	}
	return sum
}

func (c *Classifier) getAllCategories() []string {
	var keys []string
	for k := range c.CatCount {
		keys = append(keys, k)
	}
	return keys
}

func (c *Classifier) probabilityOfWordInCategory(feature string, category string) float64 {
	if c.totalCountInCategory(category) == 0 {
		return 0.0
	}
	return c.countOfWordInCategory(feature, category) / c.totalCountInCategory(category)
}

func (c *Classifier) weightedProbability(categories []string, word string, category string) float64 {
	totalCountOfWordInAllCategories := 0.0
	probability := c.probabilityOfWordInCategory(word, category)
	for _, category := range categories {
		totalCountOfWordInAllCategories += c.countOfWordInCategory(word, category)
	}
	weighedProbability := (totalCountOfWordInAllCategories * probability) / (1 + totalCountOfWordInAllCategories)
	return weighedProbability
}

func (c *Classifier) probabilityForCategory(categories []string, features []string, category string) float64 {
	totalCountInCategory := c.totalCountInCategory(category)
	docProbability := c.probabilityOfEachWordForCategory(categories, features, category)
	return docProbability * totalCountInCategory
}

func (c *Classifier) probabilityOfEachWordForCategory(categories []string, words []string, category string) float64 {
	probability := 1.0
	for _, word := range words {
		probability *= c.weightedProbability(categories, word, category)
	}
	return probability
}

func AsReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
