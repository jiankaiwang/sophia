# Tensorflow Serving with Docker



## Reference

* Tensorflow: https://www.tensorflow.org/tfx/serving/docker



## Quick Notes

The official example as the below.

```sh
# pull down Tensorflow Serving image 
docker pull tensorflow/serving

# clone the git repository 
git clone https://github.com/tensorflow/serving

# location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# start tensorflow serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &
    
# query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict
    
# returns
# {"predictions": [2.5, 3.0, 4.5]}
```

A simple python example.

```python
# prepare a simple example
my_data = {"instances": [1.0, 2.0, 5.0]}

# the serving url
serving_url = "http://localhost:8501/v1/models/half_plus_two:predict"

# a post request
r = requests.post(serving_url,
                  data=json.dumps(my_data),
                  headers={'Content-Type': 'application/octet-stream'})
print(r.status_code, r.content)
```



### Running a serving image

The serving images (both CPU and GPU) provides the following attributes.

* Port 8500 exposed for gRPC
* Port 8501 exposed for REST API
* Optional environment variable `MODEL_NAME` (default to `model`)
* Optional environment variable `MODEL_BASE_PATH` (default to `/models`)

The command:

```sh
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
  
# default command as below
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=my_model --model_base_path=/models/my_model
```

If you run via docker, three requirements is listed.

* An `open port` on your host to serve on.
* A `SavedModel` to serve.
* A `name` for your model that your client will request to.

```sh
docker run -t --rm -p 8501:8501 \
	-v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
	-e MODEL_NAME=half_plus_two \
	tensorflow/serving:latest &
```



### Creating your own serving image and Serving example

First run a base serving image as a daemon.

```sh
docker run -d --name serving_base tensorflow/serving:latest
```

Copy the SavedModel folder to the container's model folder.

```sh
docker cp <path>/<model_name> serving_base:/models/<model_name>
```

Commit the change to the container and export it as a new image.

```sh
docker commit -a "<author>" -m "ENV MODEL_NAME <model_name>" serving_base <new_image_name>
```

You can view the history of the image.

```sh
docker history <new_image_name>
```

Now you can stop and remove the base container.

```sh
docker kill serving_base
docker rm serving_base
```

Start a container based on new docker image.

```sh
docker run -t --rm -p 8501:8501 \
	-e MODEL_NAME=<model_name> \
	<new_image_name>:latest &
```

A simple call reuqests the serving model.

```sh
curl -d '{"image": [@test.png]}' -X POST http://localhost:8501/v1/models/TFE_SavedModel:predict
```















