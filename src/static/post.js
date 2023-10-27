document.addEventListener("DOMContentLoaded", function () {
    // Находим форму и кнопку
    const predictionForm = document.getElementById("prediction-form");
    const predictButton = predictionForm.querySelector("button[type=button]");
  
    // Добавляем обработчик события на кнопку "Предсказание"
    predictButton.addEventListener("click", function () {
      // Создаем объект FormData, чтобы собрать данные из формы
      const formData = new FormData(predictionForm);
  
      // Определите URL, на который вы хотите отправить POST-запрос
      const url = "/predict"; // Замените это на правильный URL
  
      // Выполняем POST-запрос с использованием fetch
      fetch(url, {
        method: "POST",
        body: formData, // Данные формы
      })
        .then((response) => response.json()) // Парсим ответ как JSON (если сервер возвращает JSON)
        .then((data) => {
          // Обработка ответа от сервера
          document.getElementById("prediction-result").textContent = data.prediction; // Замените "result" на поле, в котором сервер возвращает предсказание
        })
        .catch((error) => {
          console.error("Произошла ошибка:", error);
        });
    });
  });
  