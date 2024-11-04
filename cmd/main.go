package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/Cadeusept/bayesian-classifier/internal/clients"
	"github.com/Cadeusept/bayesian-classifier/internal/usecase/bayesian_classifier"
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

	classifier := bayesian_classifier.New()

	classifier.CalculateStatistics(data)

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
}
