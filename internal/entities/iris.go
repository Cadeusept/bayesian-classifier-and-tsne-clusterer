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
