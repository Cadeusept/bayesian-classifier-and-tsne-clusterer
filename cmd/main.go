package main

import (
	"fmt"
	"log"

	"github.com/Cadeusept/bayesian-classifier/internal/clients"
	"github.com/Cadeusept/bayesian-classifier/internal/usecase/bayesian_classifier"
)

func main() {
	data, err := clients.LoadData("iris.csv")
	if err != nil {
		log.Fatal(err)
	}

	classifier := bayesian_classifier.New()

	classifier.CalculateStatistics(data)

	sample := []float64{5.1, 3.5, 1.4, 0.2}
	predictedClass := classifier.Classify(sample)
	fmt.Printf("Предсказанный класс для образца %+v: %s\n", sample, predictedClass)
}
