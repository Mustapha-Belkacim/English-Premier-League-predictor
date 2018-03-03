# English premier leauge predictor
AI that predicts English premier league results with a nice web based interface

![alt text](predictions/static/predictions/images/screen%201.jpg)
![alt text](predictions/static/predictions/images/screen%203.png)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Dependencies

Install missing dependencies with [pip](https://pip.pypa.io/en/stable/). 

```
$ pip install -r requirements.txt
```

In case you have trouble installing xgboost this [solution](http://stackoverflow.com/a/39811079) works 

## Usage
Run `python manage.py runserver` and open your browser http://127.0.0.1:8000/ Enjoy.

if you want more explanation on the process followed :

-In terminal go under docs/ 
-Run `jupyter notebook` , then the code will pop up in your browser.
-Open the notebook that you  want and enjoy reading

Install jupyter [here](http://jupyter.readthedocs.io/en/latest/install.html).

## Built With
* [Django]() - The web framework used
* [pandas]() - Data preprocessing
* [scikit-learn]() - Machine learning
* [xgboost]() - xgboost model

## Contributing

Please read [CONTRIBUTING.md](https://github.com/Mustapha-Belkacim/English-Premier-League-predictor/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Mustapha Belkacim** - *Initial work* - [Mustapha-Belkacim](https://github.com/Mustapha-Belkacim)
* **Mustapha Belkacim** - *Initial work* - [Mustapha-Belkacim](https://github.com/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details