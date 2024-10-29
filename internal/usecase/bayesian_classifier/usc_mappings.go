package bayesian_classifier

import "gonum.org/v1/gonum/mat"

// Преобразование [][]float64 в *mat.SymDense
func toSymDense(matrix [][]float64) *mat.SymDense {
	n := len(matrix)
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sym.SetSym(i, j, matrix[i][j])
		}
	}
	return sym
}
