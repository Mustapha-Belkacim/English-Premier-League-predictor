from django.shortcuts import render
from django.views import View, generic
from .services.predictor import get_results

class Index(View):
    template_name = 'predictions/index.html'
    model = 'xgboost'
    season = '16/17'
    results = ''
    leadboard = ''

    def get(self, request):
        self.results = get_results(self.season)
        #self.results = predict_season(self.season, self.model)
        return render(request, self.template_name, {'results'  :self.results,
                                                    'leadboard':self.leadboard})

    def post(self, request):
        self.model = request.POST['model']
        self.season = request.POST['season']
        self.results = get_results(self.season)
        return render(request, self.template_name, {'results'  :self.results,
                                                    'leadboard':self.leadboard})
