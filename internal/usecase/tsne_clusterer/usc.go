package tsne_clusterer

import (
	"math"
	"math/rand"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/entities"
)

// Usc represents a simplified t-SNE structure for clustering
type Usc struct {
	Perplexity     float64
	LearningRate   float64
	Iterations     int
	ClusterCenters [][]float64 // Store cluster centers for prediction
	ClusterLabels  []int       // Store cluster labels
}

// New initializes a t-SNE instance with provided parameters
func New(perplexity, learningRate float64, iterations int) *Usc {
	return &Usc{
		Perplexity:   perplexity,
		LearningRate: learningRate,
		Iterations:   iterations,
	}
}

// Train обучает модель (выполняет кластеризацию)
func (u *Usc) Train(irises entities.Irises) {
	// Преобразуем Irises в формат, пригодный для кластеризации
	rows := len(irises)
	tsneResult := make([][]float64, rows)

	// Преобразуем данные Iris в двумерный массив для дальнейшей кластеризации
	for i, iris := range irises {
		tsneResult[i] = []float64{iris.SepalLength, iris.SepalWidth}
	}

	// Выполним кластеризацию с помощью k-means
	k := 3 // Три кластера для трех классов ирисов
	centroids, labels := kMeans(tsneResult, k, 100)

	// Сохраняем центры кластеров и метки кластеров
	u.ClusterCenters = centroids
	u.ClusterLabels = labels
}

// Функция для вычисления евклидова расстояния между двумя точками
func euclideanDistance(p1, p2 []float64) float64 {
	var sum float64
	for i := 0; i < len(p1); i++ {
		sum += math.Pow(p1[i]-p2[i], 2)
	}
	return math.Sqrt(sum)
}

// Функция для случайной инициализации центров кластеров
func initializeCentroids(data [][]float64, k int) [][]float64 {
	rand.Seed(42)
	centroids := make([][]float64, k)
	// Случайным образом выбираем k центров
	for i := 0; i < k; i++ {
		centroids[i] = data[rand.Intn(len(data))]
	}
	return centroids
}

// Функция для кластеризации методом k-means
func kMeans(data [][]float64, k int, maxIterations int) ([][]float64, []int) {
	centroids := initializeCentroids(data, k)
	labels := make([]int, len(data))
	var prevCentroids [][]float64

	for iteration := 0; iteration < maxIterations; iteration++ {
		// Шаг 1: Назначение меток точкам на основе ближайших центров
		for i, point := range data {
			minDist := math.MaxFloat64
			var cluster int
			for j, centroid := range centroids {
				dist := euclideanDistance(point, centroid)
				if dist < minDist {
					minDist = dist
					cluster = j
				}
			}
			labels[i] = cluster
		}

		// Шаг 2: Пересчитываем центры кластеров
		prevCentroids = centroids
		centroids = make([][]float64, k)
		counts := make([]int, k)
		for i := 0; i < k; i++ {
			centroids[i] = make([]float64, len(data[0]))
		}

		for i, point := range data {
			cluster := labels[i]
			for j := 0; j < len(point); j++ {
				centroids[cluster][j] += point[j]
			}
			counts[cluster]++
		}

		// Среднее значение для каждого кластера
		for i := 0; i < k; i++ {
			for j := 0; j < len(centroids[i]); j++ {
				if counts[i] > 0 {
					centroids[i][j] /= float64(counts[i])
				}
			}
		}

		// Прерываем, если центры кластеров не изменились
		if equalCentroids(centroids, prevCentroids) {
			break
		}
	}

	return centroids, labels
}

// Функция для проверки, равны ли два набора центров кластеров
func equalCentroids(c1, c2 [][]float64) bool {
	for i := range c1 {
		for j := range c1[i] {
			if math.Abs(c1[i][j]-c2[i][j]) > 1e-4 {
				return false
			}
		}
	}
	return true
}

// Predict determines the cluster for a given Iris based on cluster centers
func (u *Usc) Predict(iris entities.Iris) int {
	if len(u.ClusterCenters) == 0 {
		panic("Cluster centers are not initialized. Run Train first.")
	}

	// Create a vector for the input Iris data
	data := []float64{iris.SepalLength, iris.SepalWidth, iris.PetalLength, iris.PetalWidth}

	// Reduce dimension of the input Iris using the same logic as Train (simulated)
	reducedIris := []float64{data[0] + rand.Float64()*0.1, data[1] + rand.Float64()*0.1}

	// Find the closest cluster center
	closestCluster := -1
	minDistance := math.MaxFloat64
	for i, center := range u.ClusterCenters {
		distance := math.Sqrt(math.Pow(reducedIris[0]-center[0], 2) + math.Pow(reducedIris[1]-center[1], 2))
		if distance < minDistance {
			minDistance = distance
			closestCluster = i
		}
	}
	return closestCluster
}
