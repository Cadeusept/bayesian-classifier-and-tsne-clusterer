package tsne_clusterer

import (
	"fmt"
	"log"
	"testing"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/clients"
	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/entities"
	"github.com/stretchr/testify/suite"
)

type clustererUscSuite struct {
	suite.Suite

	testData entities.Irises
	svc      *Usc
}

func TestClustererUscSuite(t *testing.T) {
	// t.SkipNow()
	suite.Run(t, new(clustererUscSuite))
}

func (s *clustererUscSuite) SetupSuite() {
	data, err := clients.LoadAllData("../../../training_samples/")
	if err != nil {
		log.Fatal(err)
	}

	clusterer := New(30, 200, 1000)

	// Разделим данные на обучающую и тестовую выборки
	trainData := make(entities.Irises, 0)
	testData := make(entities.Irises, 0)

	for i, iris := range data {
		if i%5 == 0 {
			testData = append(testData, iris)
		} else {
			trainData = append(trainData, iris)
		}
	}

	clusterer.Train(trainData)

	s.testData = testData
	s.svc = clusterer
}

func (s *clustererUscSuite) TestPredict() {
	predictedCluster1 := s.svc.Predict(entities.Iris{
		SepalLength: 5.1,
		SepalWidth:  3.5,
		PetalLength: 1.4,
		PetalWidth:  0.2,
		Class:       entities.IrisClassSetosa,
	})
	predictedCluster2 := s.svc.Predict(entities.Iris{
		SepalLength: 5.9,
		SepalWidth:  3.0,
		PetalLength: 4.2,
		PetalWidth:  1.5,
		Class:       entities.IrisClassVersicolor})
	predictedCluster3 := s.svc.Predict(entities.Iris{
		SepalLength: 6.5,
		SepalWidth:  3.0,
		PetalLength: 5.5,
		PetalWidth:  1.8,
		Class:       entities.IrisClassVirginica})

	fmt.Println(predictedCluster1, predictedCluster2, predictedCluster3)
}

func (s *clustererUscSuite) TestCountMetrics() {

	// Сопоставление кластеров с классами
	clusterToClass := make(map[int]int) // ключ - номер кластера, значение - соответствующий класс

	// Подсчитываем количество каждого класса в каждом кластере
	clusterClassCounts := make(map[int]map[int]int)
	for _, testIris := range s.testData {
		predictedCluster := s.svc.Predict(testIris)
		actualClass := int(testIris.Class)

		if _, exists := clusterClassCounts[predictedCluster]; !exists {
			clusterClassCounts[predictedCluster] = make(map[int]int)
		}
		clusterClassCounts[predictedCluster][actualClass]++
	}

	// Определяем класс с максимальным количеством объектов в каждом кластере
	for cluster, counts := range clusterClassCounts {
		maxClass := -1
		maxCount := 0
		for class, count := range counts {
			if count > maxCount {
				maxClass = class
				maxCount = count
			}
		}
		clusterToClass[cluster] = maxClass
	}

	// Создаем матрицы подсчета для расчета метрик
	truePositives := make(map[int]int)
	falsePositives := make(map[int]int)
	falseNegatives := make(map[int]int)
	trueNegatives := make(map[int]int)

	// Тестируем модель на тестовой выборке
	for _, testIris := range s.testData {
		predictedCluster := s.svc.Predict(testIris)
		actualClass := int(testIris.Class)

		mappedClass := clusterToClass[predictedCluster]

		// Увеличиваем счетчики для расчета метрик
		for cluster := range s.svc.ClusterCenters {
			if mappedClass == actualClass && cluster == predictedCluster {
				truePositives[cluster]++
			} else if cluster == predictedCluster && mappedClass != actualClass {
				falsePositives[cluster]++
			} else if cluster != predictedCluster && mappedClass == actualClass {
				falseNegatives[cluster]++
			} else {
				trueNegatives[cluster]++
			}
		}
	}

	// Расчет метрик
	var totalPrecision, totalRecall, totalFMeasure float64
	clusterCount := len(s.svc.ClusterCenters)

	for cluster := 0; cluster < clusterCount; cluster++ {
		precision := float64(truePositives[cluster]) / float64(truePositives[cluster]+falsePositives[cluster])
		recall := float64(truePositives[cluster]) / float64(truePositives[cluster]+falseNegatives[cluster])
		fMeasure := 2 * (precision * recall) / (precision + recall)

		totalPrecision += precision
		totalRecall += recall
		totalFMeasure += fMeasure

		fmt.Printf("Cluster %d - Precision: %.2f, Recall: %.2f, F-measure: %.2f\n", cluster, precision, recall, fMeasure)
	}

	// Средние метрики
	avgPrecision := totalPrecision / float64(clusterCount)
	avgRecall := totalRecall / float64(clusterCount)
	avgFMeasure := totalFMeasure / float64(clusterCount)

	fmt.Printf("\nAverage Precision: %.2f\n", avgPrecision)
	fmt.Printf("Average Recall: %.2f\n", avgRecall)
	fmt.Printf("Average F-measure: %.2f\n", avgFMeasure)
}
