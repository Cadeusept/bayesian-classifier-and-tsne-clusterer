package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/clients"
	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/entities"
	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/usecase/bayesian_classifier"
	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/usecase/tsne_clusterer"
)

const maxBufSize = 25

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	buf := make([]byte, maxBufSize)
	scanner.Buffer(buf, maxBufSize)
	scanner.Split(bufio.ScanWords)

	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	data, err := clients.LoadAllData("../training_samples/")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Enter what you want to use:\n1 - Naive Bayesian Classifier\n2 - t-SNE Clusterer")
	scanner.Scan()
	switch scanner.Text() {
	case "1":
		err = startClassifier(scanner, writer, &data)
	case "2":
		err = startClusterer(scanner, writer, &data)
	default:
		log.Fatal("invalid input")
	}
	if err != nil {
		log.Fatal(err)
	}
}

func startClassifier(scanner *bufio.Scanner, writer *bufio.Writer, data *entities.Irises) error {
	if data == nil {
		log.Fatal("empty data")
	}

	classifier := bayesian_classifier.New()

	classifier.CalculateStatistics(*data)

	fmt.Println("Enter number of items you want to classify")

	scanner.Scan()
	n, err := strconv.Atoi(scanner.Text())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Enter %d items you want to classify\n", n)

	for i := 0; i < n; i++ {
		sample := make([]float64, 4)
		for j := 0; j < 4; j++ {
			scanner.Scan()
			f, err := strconv.ParseFloat(scanner.Text(), 64)
			if err != nil {
				log.Fatal(err)
			}
			sample[j] = f
		}

		predictedClass, err := clients.MapIrisClassToStr(classifier.Classify(sample))
		if err != nil {
			log.Fatal(err)
		}
		writer.WriteString(fmt.Sprintf("Predicted class for sample %+v: %s\n", sample, predictedClass))
	}

	return nil
}

func startClusterer(scanner *bufio.Scanner, writer *bufio.Writer, data *entities.Irises) error {
	if data == nil {
		log.Fatal("empty data")
	}

	clusterer := tsne_clusterer.New(2, 1.0, 300, 0.1) // 2D t-SNE with sigma=1, 300 iterations, learning rate=0.1

	clusterer.Train(data.ToMatrix()) // Assuming ToMatrix converts data into [][]float64

	fmt.Println("Enter number of items you want to predict cluster for")
	scanner.Scan()
	n, err := strconv.Atoi(scanner.Text())
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Enter %d items you want to predict cluster for\n", n)

	for i := 0; i < n; i++ {
		sample := make([]float64, 4)
		for j := 0; j < 4; j++ {
			scanner.Scan()
			f, err := strconv.ParseFloat(scanner.Text(), 64)
			if err != nil {
				log.Fatal(err)
			}
			sample[j] = f
		}

		predictedClass := clusterer.Predict(sample) // No need to check for error here
		writer.WriteString(fmt.Sprintf("Predicted cluster for sample %+v: %d\n", sample, predictedClass))
	}

	return nil
}
