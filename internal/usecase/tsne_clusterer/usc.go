package tsne_clusterer

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal"
)

type Usc struct {
	dims         int
	sigma        float64
	iterations   int
	learningRate float64
	lowDimData   *mat.Dense
}

func New(dims int, sigma float64, iterations int, learningRate float64) *Usc {
	return &Usc{
		dims:         dims,
		sigma:        sigma,
		iterations:   iterations,
		learningRate: learningRate,
	}
}

// Train выполняет обучение t-SNE на входных данных
func (c *Usc) Train(data [][]float64) {
	matDense := internal.ToMatDense(data)

	n, _ := matDense.Dims()

	// Вычисляем вероятности в высокоразмерном пространстве
	P := computeHighDimAffinities(matDense, c.sigma)

	// Инициализируем точки в низкоразмерном пространстве случайными значениями
	Y := initializeLowDimCoords(n, c.dims)

	// Итерации градиентного спуска
	for iter := 0; iter < c.iterations; iter++ {
		Q, distances := computeLowDimAffinities(Y)
		gradient := computeGradient(P, Q, Y, distances)

		// Обновляем координаты Y
		for i := 0; i < n; i++ {
			for d := 0; d < c.dims; d++ {
				Y.Set(i, d, Y.At(i, d)-c.learningRate*gradient.At(i, d))
			}
		}
		if iter%10 == 0 {
			fmt.Printf("Iteration %d/%d completed\n", iter, c.iterations)
		}
	}

	c.lowDimData = Y
}

// Predict определяет ближайший кластер для нового образца
func (c *Usc) Predict(sample []float64) int {
	if c.lowDimData == nil {
		panic("Clusterer not yet fitted with data")
	}

	// Количество ближайших соседей
	k := 5 // Вы можете настроить это значение

	// Находим k ближайших соседей в исходном высокоразмерном пространстве
	nearestNeighbors := findNearestNeighbors(sample, c.lowDimData, k)

	// Вычисляем среднюю позицию соседей в 2D пространстве
	average2DPosition := c.computeAveragePosition(nearestNeighbors)

	// Находим ближайшую точку в низкоразмерном пространстве
	closestCluster := c.findClosestCluster(average2DPosition)

	return closestCluster
}

// findNearestNeighbors находит k ближайших соседей образца в высокоразмерном пространстве
func findNearestNeighbors(sample []float64, lowDimData *mat.Dense, k int) []int {
	n, _ := lowDimData.Dims()
	distances := make([]float64, n)

	// Вычисляем евклидово расстояние до каждой точки в высокоразмерном пространстве
	for i := 0; i < n; i++ {
		point := lowDimData.RowView(i)
		dist := euclideanDistSquared(mat.NewVecDense(len(sample), sample), point)
		distances[i] = dist
	}

	// Сортируем расстояния и получаем индексы k ближайших соседей
	neighbors := make([]int, k)
	copy(neighbors, indicesOfSmallest(distances, k))
	return neighbors
}

// computeAveragePosition вычисляет среднюю 2D позицию ближайших соседей
func (c *Usc) computeAveragePosition(neighbors []int) []float64 {
	n := len(neighbors)
	if n == 0 {
		return nil
	}

	var avgX, avgY float64
	for _, idx := range neighbors {
		avgX += c.lowDimData.At(idx, 0)
		avgY += c.lowDimData.At(idx, 1)
	}

	return []float64{avgX / float64(n), avgY / float64(n)}
}

// findClosestCluster находит ближайший кластер к средней позиции в низкоразмерном пространстве
func (c *Usc) findClosestCluster(averagePosition []float64) int {
	n, _ := c.lowDimData.Dims()
	minDist := math.MaxFloat64
	closestCluster := -1

	for i := 0; i < n; i++ {
		currentPoint := c.lowDimData.RowView(i)
		dist := euclideanDistSquared(mat.NewVecDense(len(averagePosition), averagePosition), currentPoint)
		if dist < minDist {
			minDist = dist
			closestCluster = i
		}
	}

	return closestCluster
}

// indicesOfSmallest возвращает индексы k минимальных элементов в массиве
func indicesOfSmallest(arr []float64, k int) []int {
	indices := make([]int, len(arr))
	for i := range arr {
		indices[i] = i
	}
	// Сортируем индексы в зависимости от соответствующих значений в arr
	sort.SliceStable(indices, func(i, j int) bool {
		return arr[indices[i]] < arr[indices[j]]
	})
	return indices[:k]
}

// computeHighDimAffinities вычисляет вероятности P для высокоразмерного пространства
func computeHighDimAffinities(data *mat.Dense, sigma float64) *mat.Dense {
	n, _ := data.Dims()
	P := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				ai := data.RowView(i)
				aj := data.RowView(j)
				dist := euclideanDistSquared(ai, aj)
				P.Set(i, j, math.Exp(-dist/(2*sigma*sigma)))
			}
		}
		row := mat.Row(nil, i, P)
		P.SetRow(i, normalizeRow(mat.NewVecDense(len(row), row)).RawVector().Data)
	}

	return P
}

// initializeLowDimCoords инициализирует точки в низкоразмерном пространстве случайными значениями
func initializeLowDimCoords(n, dims int) *mat.Dense {
	Y := mat.NewDense(n, dims, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < dims; j++ {
			Y.Set(i, j, rand.Float64()*2-1) // Случайные значения от -1 до 1
		}
	}
	return Y
}

// computeLowDimAffinities вычисляет вероятности Q для низкоразмерного пространства
func computeLowDimAffinities(Y *mat.Dense) (*mat.Dense, *mat.Dense) {
	n, _ := Y.Dims()
	Q := mat.NewDense(n, n, nil)
	distances := mat.NewDense(n, n, nil)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				ai := Y.RowView(i)
				aj := Y.RowView(j)
				dist := 1.0 / (1.0 + euclideanDistSquared(ai, aj))
				distances.Set(i, j, dist)
				Q.Set(i, j, dist)
			}
		}
		row := mat.Row(nil, i, Q)
		Q.SetRow(i, normalizeRow(mat.NewVecDense(len(row), row)).RawVector().Data)
	}

	return Q, distances
}

// computeGradient вычисляет градиент для обновления координат Y
func computeGradient(P, Q, Y, distances *mat.Dense) *mat.Dense {
	n, dims := Y.Dims()
	gradient := mat.NewDense(n, dims, nil)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				diff := make([]float64, dims)
				for d := 0; d < dims; d++ {
					diff[d] = Y.At(i, d) - Y.At(j, d)
				}
				factor := 4 * (P.At(i, j) - Q.At(i, j)) * distances.At(i, j)
				for d := 0; d < dims; d++ {
					gradient.Set(i, d, gradient.At(i, d)+factor*diff[d])
				}
			}
		}
	}
	return gradient
}

// normalizeRow нормализует строку до суммы 1
func normalizeRow(row *mat.VecDense) *mat.VecDense {
	sum := mat.Sum(row)
	normRow := mat.NewVecDense(row.Len(), nil)
	normRow.ScaleVec(1/sum, row)
	return normRow
}

func euclideanDistSquared(a, b mat.Vector) float64 {
	n := a.Len()
	var sum float64
	for i := 0; i < n; i++ {
		diff := a.AtVec(i) - b.AtVec(i)
		sum += diff * diff
	}
	return sum
}
