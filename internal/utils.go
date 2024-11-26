package internal

import (
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// Преобразование [][]float64 в *mat.SymDense
func ToSymDense(matrix [][]float64) *mat.SymDense {
	n := len(matrix)
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, matrix[i][j])
		}
	}
	return sym
}

// Преобразование [][]float64 в *mat.Dense
func ToMatDense(data [][]float64) *mat.Dense {
	rows := len(data)
	cols := len(data[0])
	flatData := make([]float64, 0, rows*cols)

	for _, row := range data {
		flatData = append(flatData, row...)
	}

	return mat.NewDense(rows, cols, flatData)
}

// Функция для проверки, равны ли два набора центров кластеров
func EqualCentroids(c1, c2 [][]float64) bool {
	for i := range c1 {
		for j := range c1[i] {
			if math.Abs(c1[i][j]-c2[i][j]) > 1e-4 {
				return false
			}
		}
	}
	return true
}

// Функция для вычисления евклидова расстояния между двумя точками
func EuclideanDistance(p1, p2 []float64) float64 {
	var sum float64
	for i := 0; i < len(p1); i++ {
		sum += math.Pow(p1[i]-p2[i], 2)
	}
	return math.Sqrt(sum)
}

// Функция для случайной инициализации центров кластеров
func InitializeCentroids(data [][]float64, k int) [][]float64 {
	centroids := make([][]float64, k)
	// Случайным образом выбираем k центров
	for i := 0; i < k; i++ {
		centroids[i] = data[rand.Intn(len(data))]
	}
	return centroids
}
