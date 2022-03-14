Pour tester la solution en local

pip install pandas
pip install Flask   
pip install flask-dropzone

lancer le serveur flask run main.py

Tester une requéte JSON tel que :

127.0.0.1:5000/integrity

{
	"manufacturer_name":"Renault",
	"model_name":"Megane",
	"transmission":"mechanical",
	"color":"black",
	"odometer_value":280000,
	"year":2010,
	"engine_fuel":"diesel",
	"engine_type":"diesel",
	"price":6999.0
}