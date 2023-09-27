from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from movie.models import Movie
import os
import numpy as np

import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv, find_dotenv

# Create your views here.
def recommendations(request):
    searchTerm = request.GET.get('recommendMovie')

    if searchTerm:
        _ = load_dotenv('../openAI.env')
        openai.api_key = os.environ['openAI_api_key']
        items = Movie.objects.all()

        req = searchTerm
        emb_req = get_embedding(req,engine='text-embedding-ada-002')

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)

        if req: 
            movies = Movie.objects.filter(title__icontains=items[idx].title) 
        else: 
            movies = Movie.objects.all()
        return render(request, 'recommendations.html', {'searchTerm':searchTerm, 'movies': movies})
    else:
        movies = Movie.objects.all()
        return render(request, 'recommendations.html', {'searchTerm':searchTerm, 'movies': movies})
    
