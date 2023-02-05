package naive

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"testing"
	"time"
)

var categories = []string{
	"Dog", "Cat",
}

func TestSimpleClassifier(t *testing.T) {
	classifier := New()

	classifier.TrainString("German Shepherd", "Dog")
	classifier.TrainString("Pointer", "Dog")
	classifier.TrainString("Black kitty", "Cat")
	classifier.TrainString("White kitten", "Cat")
	classifier.TrainString("White kitty", "Cat")
	classifier.TrainString("Guppy kitty", "Fish")
	classifier.TrainString("Guppy king", "Fish")

	probabilities, topResult := classifier.Probabilities("Kitty white")
	fmt.Println(topResult)
	fmt.Println(probabilities)
}

// https://medium.com/analytics-vidhya/how-naive-bayes-algorithm-work-d53e0a13a364
func TestWeatherClassifier(t *testing.T) {
	classifier := New()

	classifier.TrainString("Sunny", "No")
	classifier.TrainString("Sunny", "No")
	classifier.TrainString("Overcast", "Yes")
	classifier.TrainString("Rainy", "Yes")
	classifier.TrainString("Rainy", "Yes")
	classifier.TrainString("Rainy", "No")
	classifier.TrainString("Overcast", "Yes")
	classifier.TrainString("Sunny", "No")
	classifier.TrainString("Sunny", "Yes")
	classifier.TrainString("Rainy", "Yes")
	classifier.TrainString("Sunny", "Yes")
	classifier.TrainString("Overcast", "Yes")
	classifier.TrainString("Overcast", "Yes")
	classifier.TrainString("Rainy", "No")

	probabilities, topResult := classifier.Probabilities("Overcast")
	fmt.Println(topResult)
	fmt.Println(probabilities)
}

func TestClassifier(t *testing.T) {
	f, err := os.Open("./classification_training_data.csv")

	if err != nil {
		panic(err)
	}

	// Train on CSV data
	classifier := New()
	r := csv.NewReader(f)

	for {
		record, err := r.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			panic(err)
		}

		classifier.TrainString(record[0], record[1])
	}

	now := time.Now()
	probabilities, topResult := classifier.Probabilities("Veldskoen")

	fmt.Println("Calculation took: ", time.Now().Sub(now))
	fmt.Printf("%s: %f", topResult, probabilities[topResult])
	fmt.Println(probabilities)
}
