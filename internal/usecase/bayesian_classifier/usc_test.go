package bayesian_classifier

import (
	"log"
	"testing"

	"github.com/Cadeusept/bayesian-classifier/internal/clients"
	"github.com/stretchr/testify/suite"
)

type bayesianClassifierUscSuite struct {
	suite.Suite

	svc *Usc
}

func TestBayesianClassifierUscSuite(t *testing.T) {
	t.SkipNow()
	suite.Run(t, new(bayesianClassifierUscSuite))
}

func (s *bayesianClassifierUscSuite) SetupSuite() {
	data, err := clients.LoadAllData("../../../training_samples/")
	if err != nil {
		log.Fatal(err)
	}

	classifier := New()

	classifier.CalculateStatistics(data)

	s.svc = classifier
}

func (s *bayesianClassifierUscSuite) TestClassify() {
	predictedSetosaClass := s.svc.Classify([]float64{5.1, 3.5, 1.4, 0.2})
	s.Suite.Require().EqualValues(0, predictedSetosaClass)

	predictedVersicolorClass := s.svc.Classify([]float64{5.9, 3.0, 4.2, 1.5})
	s.Suite.Require().EqualValues(1, predictedVersicolorClass)

	predictedVirginicaClass := s.svc.Classify([]float64{6.5, 3.0, 5.5, 1.8})
	s.Suite.Require().EqualValues(2, predictedVirginicaClass)
}
