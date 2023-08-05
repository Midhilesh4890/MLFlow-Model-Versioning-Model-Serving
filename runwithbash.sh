curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 10,
  "sepal_width": 110,
  "petal_length": 1110,
  "petal_width": 11110
}'