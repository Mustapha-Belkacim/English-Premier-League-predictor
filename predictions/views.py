from django.shortcuts import render
from django.views import View, generic


class Index(View):
    template_name = 'predictions/index.html'
    model = 'xgboost'
    season = '16/17'
    results = 'kfdgkfd'
    leadboard = 'dfg'

    def get(self, request):
        return render(request, self.template_name, {'results'  :self.results,
                                                    'leadboard':self.leadboard})

    def post(self, request):
        self.model = request.POST['model']
        self.season = request.POST['season']
        return render(request, self.template_name, {'results'  :self.results,
                                                    'leadboard':self.leadboard})
