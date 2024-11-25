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

	svc *Usc
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

	clusterer.Train(data)

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
