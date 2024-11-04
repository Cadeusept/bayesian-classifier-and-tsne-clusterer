package main

import (
	"fmt"
	"log"

	"github.com/Cadeusept/bayesian-classifier/internal/clients"
	"github.com/Cadeusept/bayesian-classifier/internal/usecase/bayesian_classifier"
)

func main() {
	data, err := clients.LoadAllData("../training_samples/")
	if err != nil {
		log.Fatal(err)
	}

	classifier := bayesian_classifier.New()

	classifier.CalculateStatistics(data)

	sample := []float64{5.1, 3.5, 1.4, 0.2}
	predictedClass, err := clients.MapIrisClassToStr(classifier.Classify(sample))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Предсказанный класс для образца %+v: %s\n", sample, predictedClass)
}
