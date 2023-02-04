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

func TestClassifierSimple(t *testing.T) {
	classifier := New()

	classifier.TrainString("German Shepherd", "Dog")
	classifier.TrainString("Pointer", "Dog")
	classifier.TrainString("Black kitty", "Cat")
	classifier.TrainString("White kitten", "Cat")

	probabilities, topResult := classifier.Probabilities("Kitty")
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
	probabilities, topResult := classifier.Probabilities("Trade Professional - TP 6000 4S - 5.5kW 13HP 6.8kVA")

	fmt.Println("Calculation took: ", time.Now().Sub(now))
	fmt.Println(topResult)
	fmt.Println(probabilities)
}
