package internal

import "gonum.org/v1/gonum/mat"

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
