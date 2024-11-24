package tsne_clusterer

import (
	"fmt"
	"log"
	"testing"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/clients"
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

	matrix := data.ToMatrix()

	clusterer := New(4, 0.1, 300, 0.01)

	clusterer.Train(matrix)

	s.svc = clusterer
}

func (s *clustererUscSuite) TestPredict() {
	predictedCluster1 := s.svc.Predict([]float64{5.1, 3.5, 1.4, 0.2})
	predictedCluster2 := s.svc.Predict([]float64{5.9, 3.0, 4.2, 1.5})
	predictedCluster3 := s.svc.Predict([]float64{6.5, 3.0, 5.5, 1.8})

	fmt.Println(predictedCluster1, predictedCluster2, predictedCluster3)
}
