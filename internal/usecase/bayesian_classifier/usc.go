package bayesian_classifier

import (
	"log"

	"gonum.org/v1/gonum/stat/distmv"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal"
	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/entities"
)

type Usc struct {
	ClassMeans       map[entities.IrisClass][]float64
	ClassCovariances map[entities.IrisClass][][]float64
	ClassCounts      map[entities.IrisClass]int
}

func New() *Usc {
	return &Usc{
		ClassMeans:       make(map[entities.IrisClass][]float64),
		ClassCovariances: make(map[entities.IrisClass][][]float64),
		ClassCounts:      make(map[entities.IrisClass]int),
	}
}

// Функция для вычисления среднего и дисперсии для каждого класса
func (usc *Usc) CalculateStatistics(data []entities.Iris) {
	for _, sample := range data {
		class := sample.Class
		usc.ClassCounts[class]++

		features := []float64{sample.SepalLength, sample.SepalWidth, sample.PetalLength, sample.PetalWidth}
		if _, exists := usc.ClassMeans[class]; !exists {
			usc.ClassMeans[class] = make([]float64, 4)
			usc.ClassCovariances[class] = make([][]float64, 4)
			for i := range usc.ClassCovariances[class] {
				usc.ClassCovariances[class][i] = make([]float64, 4)
			}
		}
		for i := range features {
			usc.ClassMeans[class][i] += features[i]
		}
	}

	// Рассчитываем среднее
	for class, mean := range usc.ClassMeans {
		count := float64(usc.ClassCounts[class])
		for i := range mean {
			mean[i] /= count
		}
	}

	// Рассчитываем ковариации
	for _, sample := range data {
		class := sample.Class
		features := []float64{sample.SepalLength, sample.SepalWidth, sample.PetalLength, sample.PetalWidth}
		for i := range features {
			for j := range features {
				usc.ClassCovariances[class][i][j] += (features[i] - usc.ClassMeans[class][i]) * (features[j] - usc.ClassMeans[class][j])
			}
		}
	}
	for class, cov := range usc.ClassCovariances {
		count := float64(usc.ClassCounts[class] - 1)
		for i := range cov {
			for j := range cov[i] {
				cov[i][j] /= count
			}
		}
	}
}

// Функция для классификации нового образца
func (usc *Usc) Classify(sample []float64) entities.IrisClass {
	var bestClass entities.IrisClass
	var bestProb float64
	totalCount := 0
	for _, count := range usc.ClassCounts {
		totalCount += count
	}

	for class, mean := range usc.ClassMeans {
		dist, ok := distmv.NewNormal(mean, internal.ToSymDense(usc.ClassCovariances[class]), nil)
		if !ok {
			log.Fatal("Ошибка создания распределения для класса", class)
		}

		prob := float64(usc.ClassCounts[class]) / float64(totalCount) * dist.Prob(sample)
		if prob > bestProb {
			bestProb = prob
			bestClass = class
		}
	}

	return bestClass
}
