package clients

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/Cadeusept/bayesian-classifier-and-tsne-clusterer/internal/entities"
)

func LoadAllData(path string) (entities.Irises, error) {
	irises := make(entities.Irises, 0, 150)
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	for _, e := range entries {
		if e.IsDir() {
			newIrises, err := LoadAllData(path + fmt.Sprintf("%s/", e.Name()))
			if err != nil {
				return nil, err
			}

			irises = append(irises, newIrises...)
		} else if filename := e.Name(); filename[len(filename)-4:] == ".csv" {
			newIrises, err := LoadData(fmt.Sprintf("%s/%s", path, filename))
			if err != nil {
				return nil, err
			}

			irises = append(irises, newIrises...)
		}
	}

	return irises, nil
}

// Функция для чтения данных из CSV файла
func LoadData(filename string) (entities.Irises, error) {
	file, err := os.Open(fmt.Sprintf("%s", filename))
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
		class, err := MapStrToIrisClass(row[4])
		if err != nil {
			return nil, err
		}
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

func MapStrToIrisClass(s string) (entities.IrisClass, error) {
	switch s {
	case "setosa":
		return 0, nil
	case "versicolor":
		return 1, nil
	case "virginica":
		return 2, nil
	default:
		return -1, fmt.Errorf("error mapping string to iris class: undefined class")
	}
}

func MapIrisClassToStr(c entities.IrisClass) (string, error) {
	switch c {
	case 0:
		return "setosa", nil
	case 1:
		return "versicolor", nil
	case 2:
		return "virginica", nil
	default:
		return "", fmt.Errorf("error mapping iris class to string: undefined class")
	}
}
