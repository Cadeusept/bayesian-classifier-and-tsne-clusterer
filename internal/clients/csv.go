package clients

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/Cadeusept/bayesian-classifier/internal/entities"
)

// Функция для чтения данных из CSV файла
func LoadData(filename string) (entities.Irises, error) {
	file, err := os.Open(fmt.Sprintf("../training_samples/%s", filename))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1 // Для обработки строк с разным количеством полей
	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data entities.Irises
	for _, row := range rawData[1:] { // Пропускаем заголовок
		sepalLength, _ := strconv.ParseFloat(row[0], 64)
		sepalWidth, _ := strconv.ParseFloat(row[1], 64)
		petalLength, _ := strconv.ParseFloat(row[2], 64)
		petalWidth, _ := strconv.ParseFloat(row[3], 64)
		class := entities.IrisClass(row[4])
		data = append(data, entities.Iris{
			SepalLength: sepalLength,
			SepalWidth:  sepalWidth,
			PetalLength: petalLength,
			PetalWidth:  petalWidth,
			Class:       class,
		})
	}
	return data, nil
}
