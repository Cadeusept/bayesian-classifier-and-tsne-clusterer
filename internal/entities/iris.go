package entities

type IrisClass string

const (
	IrisClassSetosa     IrisClass = "setosa"
	IrisClassVersicolor IrisClass = "versicolor"
	IrisClassVirginica  IrisClass = "virginica"
)

type Irises []Iris

type Iris struct {
	SepalLength float64
	SepalWidth  float64
	PetalLength float64
	PetalWidth  float64
	Class       IrisClass
}
