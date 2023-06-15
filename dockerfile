FROM python:3.9.5
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "--server.port", "8501", "app.py"]


# generate image using : docker build -t ai-lab-final-mini-app .
#docker run -d -p 8501:8501 ai-lab-final-mini-app


# docker tag ai-lab-final-mini-app:latest ksreey/ai-lab-final-mini-app:latest
# docker push ksreey/ai-lab-final-mini-app:latest


# look at image using docker images
# to deploy create a yaml file and remember to give the same image name there
# kubectl delete deployment "deployment name" : to delete the deployment

# to deploy :  kubectl apply -f deployment.yaml

# minikube dashboard : to open the dashboard and we can see the pods

#kubectl get pods : to see the pods

# kubectl expose deployment ai-mini-app --type=NodePort --port=8501 : to expose the deployment

# kubectl get services : to see the services

# kubectl get deployments : to see the deployments

# minikube service ai-mini-app : to open the service


# minikube service "service name" in this example its ai-mini-app

# need to install docker desktop, kubernetes CLI and minikube