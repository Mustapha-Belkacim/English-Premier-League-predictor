# English premier leauge predictor
AI that predicts English premier league results with **80%** accuracy and a nice web based interface

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

if you want more explanation on the process followed building the core code:

-In terminal go under docs/ 

-Run `jupyter notebook` , then the code will pop up in your browser.

-Open the notebook that you want and enjoy reading

Install jupyter [here](http://jupyter.readthedocs.io/en/latest/install.html).

## Built With
* [Django](https://www.djangoproject.com/) - The web framework used
* [pandas](https://pandas.pydata.org/) - Data preprocessing
* [scikit-learn](http://scikit-learn.org/stable/) - Machine learning
* [xgboost](https://github.com/dmlc/xgboost) - xgboost model

## Contributing

Please read [CONTRIBUTING.md](https://github.com/Mustapha-Belkacim/English-Premier-League-predictor/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Mustapha Belkacim** - [Mustapha-Belkacim](https://github.com/Mustapha-Belkacim)
* **Hajar Zerouani** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details