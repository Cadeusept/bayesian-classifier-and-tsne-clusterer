package entities

type IrisClass int

const (
	IrisClassSetosa     IrisClass = 0
	IrisClassVersicolor IrisClass = iota
	IrisClassVirginica  IrisClass = iota
)

type Irises []Iris

type Iris struct {
	SepalLength float64
	SepalWidth  float64
	PetalLength float64
	PetalWidth  float64
	Class       IrisClass
}

// ToMatrix превращает Irises в 2D матрицу
func (irises Irises) ToMatrix() [][]float64 {
	matrix := make([][]float64, len(irises))
	for i, iris := range irises {
		matrix[i] = []float64{iris.SepalLength, iris.SepalWidth, iris.PetalLength, iris.PetalWidth}
	}
	return matrix
}
