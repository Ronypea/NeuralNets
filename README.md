# NeuralNets
Applied math lab course 3 ITMO

## Описание задачи 
С помощью моделирования искусственной нейронной сети необходимо решить две задачи.

Необходимо написать модель сети самостоятельно,
также самостоятельно запрограммировать оптимизацию (это может быть, например,
алгоритм обратного распространения ошибки, градиентный спуск и т.д.). Для тренировки
сети используйте самостоятельно сформированный датасет (размерностью, например, от
100 датапоинтов и выше для первой функции).

**Вход**: аргументы функции. **Выход**: результат функции

![image](https://user-images.githubusercontent.com/29158476/60629466-947d3b80-9dfe-11e9-891a-d6e815017d1e.png)

## Реализация 

В качестве активационной функции для первой задачи была выбрана **логистическая функция**, для второй - **пороговая**.
Оптимизация - **метод градиентного спуска**, обратное распространие ошибки
