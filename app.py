from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix


# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.secret_key = 'S1122P'  # Replace with a secure key

# Function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to process uploaded file (replace with your actual logic)
def process_file(filepath):
    # Perform your file processing logic here
    # For example, read the file, perform calculations, etc.
    # This is a placeholder, replace it with your actual code
    result = f"File processed successfully! Filepath: {filepath}"
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file is selected
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('upload_file'))

        file = request.files['file']
        # Check if file is empty
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('upload_file'))

        if file and allowed_file(file.filename):
            # Secure filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            # Save file
            file.save(filepath)

            # Process uploaded file
            result = analyze(filepath)

            # Redirect to result page with result message
            return redirect(url_for('show_result', result=result))

        else:
            flash('Allowed file types are: ' + ', '.join(ALLOWED_EXTENSIONS))
            return redirect(url_for('upload_file'))

    return '''
            <!doctype html>
        <html lang="en">

        <head>
            <title>Home</title>
            <!-- Required meta tags -->
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
            <link rel="stylesheet" href="index.css">
            <!-- Bootstrap CSS v5.2.1 -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"
                integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous" />
            <link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/core.css" />
        </head>

        <body>
            <header style="height: 8dvh; background: #52796f; " class="shadow-sm p-3 mb-5">
            <h2 class="text-white text-center"><b>Human Scream Detection</b></h2>
            </header>
            <main class="container">

                <section class="">
                    <div class="card text-start mt-3 shadow-sm p-3 mb-5 bg-body-tertiary rounded">
                        <div class="card-body">
                            <h1 class="card-header pb-5 text-center">Upload File</h1>
                            <form method="post" class="d-grid p-4 gap-4" enctype="multipart/form-data">
                                <input type="file" name="file">
                                <input type="submit" value="Upload" class="btn btn-success">
                            </form>
                        </div>
                    </div>


                </section>
            </main>
            <footer>
                <!-- place footer here -->
            </footer>
            <!-- Bootstrap JavaScript Libraries -->
            <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js"
                integrity="sha384-I7E8VVD/ismYTF4hNIPjVp/Zjvgyol6VFvRkX/vR+Vc4jQkC+hVqc2pM8ODewa9r"
                crossorigin="anonymous"></script>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.min.js"
                integrity="sha384-BBtl+eGJRgqQAUMxJ7pMwbEyER4l1g+O15P+16Ep7Q9Q+zqX6gSbd85u4mG4QzX+"
                crossorigin="anonymous"></script>
            <script type="module" src="https://pyscript.net/releases/2024.1.1/core.js"></script>

        </body>

        </html>
            '''

@app.route('/result/<result>')
def show_result(result):
    
    if(result[1] == '0'):
        return''' 
            <!doctype html>
            <html lang="en">

            <head>
                <title>Result</title>
                <!-- Required meta tags -->
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />

                <!-- Bootstrap CSS v5.2.1 -->
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"
                    integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous" />
            </head>

            <body>
                <header style="height: 8dvh; background: #52796f; " class="shadow-sm p-3 mb-5">
                    <h2 class="text-white text-center"><b>Human Scream Detection</b></h2>
                </header>
                <main class="container ">
                    <section>
                    
                        <div class="card mt-2">
                            <div class="card-body">
                                <h4 class="card-title">No scream detected, No immediate action required</h4>
                            </div>
                        </div>
                    
                    </section>
                </main>


            </body>

            </html>
        '''
    
    else:
        return''' 
            <!doctype html>
            <html lang="en">

            <head>
                <title>Result</title>
                <!-- Required meta tags -->
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />

                 <!-- Bootstrap CSS v5.2.1 -->
                 <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"
                     integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous" />
             </head>

             <body>
                 <header style="height: 8dvh; background: #52796f; " class="shadow-sm p-3 mb-5">
                     <h2 class="text-white text-center"><b>Human Scream Detection</b></h2>
                 </header>
                 <main class="container ">
                     <section>
                    
                         <div class="card text-start mt-2">
                             <div class="card-body">
                             <h4 class="card-title">Potential scream detected, Crime analysis and intervention required</h4>
                             </div>
                         </div>
                   
                     </section>
                 </main>


             </body>

             </html>
         '''



def extract_features(file_path):
    sample_rate, audio_data= wavfile.read(file_path) 
    # print(len(audio_data) )
    # print(sample_rate, audio_data) 
    mfcc_features = np.mean(librosa.feature.mfcc(y=audio_data.astype(float),sr=sample_rate,n_mfcc=23),axis=1) 
    return mfcc_features




def load_data(data_dir):
    features=[]
    labels=[]
    
    for label in os.listdir(data_dir): 
       label_path=os.path.join(data_dir,label)
       
       if os.path.isdir(label_path):
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path,filename)
            if filename.endswith(".wav"):
                 features.append(extract_features(file_path))
                 labels.append(label)
    return np.array(features) ,np.array(labels)                
             



def evaluate_model(X_train,X_test,y_train,y_test):
    classifiers=[LogisticRegression(max_iter=500,random_state=0),
                 SVC(random_state=0),
                 KNeighborsClassifier(n_neighbors=5),
                 MLPClassifier(hidden_layer_sizes=(10,8),max_iter=500,random_state=0)]
    
    acc=[]
    models=[]

    for classifier in classifiers:
        classifier.fit(X_train,y_train)
        models.append(classifier)
        
        y_pred = classifier.predict(X_test)
        
        accuracy= accuracy_score(y_test,y_pred)
        acc.append(accuracy)
        
        print(f"For {classifier}:")
        print(f"Accuracy {accuracy*100 :.2f}%")
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        # print()
        
    best_model=models[np.argmax(np.array(acc))] 
    return best_model



def analyze(file_name):
    data_dir="Audios" 
    features,labels=load_data(data_dir)
    labels[labels == 'positive'] = 1
    labels[labels == 'negative'] = 0
    labels= labels.astype(int)

   # print(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42) 
    
    scaler=StandardScaler() # x = (x - mean_of_x)/ std_of_x
    
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    best_model= evaluate_model(X_train,X_test,y_train,y_test)
    
    print("Best Model :",best_model)
    
    file_path=file_name
    data=extract_features(file_path)
    data=scaler.transform(data.reshape(1,-1))
    return best_model.predict(data)

