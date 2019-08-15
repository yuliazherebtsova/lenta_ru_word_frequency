# Задача: Тренировка навыков Python

* Необходимо написать программу на языке Python, которая выгружает минимум 10 страниц-новостей сайта https://lenta.ru в каждой категории (страницы категорий можно забить руками, страницы новостей руками не забивать):

		Россия
		Мир
		Экономика
		Силовые структуры
 		Наука и техника
 		Культура
 		Спорт
 		Интернет и СМИ
 		Путешествия
		
* Выдираем оттуда только текст статьи (без html-тегов, оглавлений и текста меню).

* Выгружаем CSV-файлы (по одному на каждую категорию), где будет топ-20 слов по популярности, представленные в 2 столбца:
		
		Слово
		Частотность

* Подсказки: выгрузить страницу по ссылке можно с помощью библиотеки requests, выделить html можно с помощью библиотеки beautifulsoup.

* Решиение: залить на github в отдельный репозиторий.
