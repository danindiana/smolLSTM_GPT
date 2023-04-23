//Generate an simple LSTM in C++.       
//Author: Dan_Indiana
//Date: 04/23/2023
//Version: 1.0
//License: MIT
//Description: This program generates a simple LSTM in C++.

#include <iostream>
#include <vector>
#include <random>
#include <string>

using namespace std;

class LSTM {
     // four matrices for the weights of the model 
    vector< vector<double> > Wf;  // weights for forget gate 
    vector< vector<double> > Wi;  // weights for input gate 
    vector< vector<double> > Wo;  // weights for output gate 
    vector< vector<double> > Wc;  // weights for cell state 

    // four bias vectors, one for each gate 
    vector<double> bf;   // bias for forget gate 
    vector<double> bi;   // bias for input gate 
    vector<double> bo;   // bias for output gate  
    vector<double> bc;   // bias for cell state

     int num_inputs;      // number of inputs to the network (length of input sequence)     

     public:                 

        LSTM(int num_inputs);

        void InitWeights();

        void ForwardPropogation(vector<double>& x);

        void BackPropogation(vector<vector<double>>& delta);     

        void UpdateWeights();                          

 };      

 LSTM::LSTM(int num_inputs){

     this->num_inputs = num_inputs;        

     Wf.resize(num_inputs,vector<double>(num_inputs));         
     Wi.resize(num_inputs,vector<double>(num_inputs));         
     Wo.resize(num_inputs,vector<double>(num_inputs));         
     Wc.resize(num_inputs,vector<double>(num_inputs));           

     bf.resize(num_inputs);          
     bi.resize(num_inputs);           
     bo.resize(num_inputs);           
     bc.resize(num_inputs);        
 }     

 void LSTM::InitWeights(){

     // Initialize weights and biases to random values 
     random_device rd;  
     mt19937 gen(rd()); 
     uniform_real_distribution<> dis(-1, 1);

     for (int i=0; i<num_inputs; i++){
         for (int j=0; j<num_inputs; j++){             

             Wf[i][j] = dis(gen);          
             Wi[i][j] = dis(gen);          
             Wo[i][j] = dis(gen);          
             Wc[i][j] = dis(gen);                     
         } 

         bf[i] = dis(gen);              
         bi[i] = dis(gen);             
         bo[i] = dis(gen);              
         bc[i] = dis(gen);   
     }       

 }     

 void LSTM::ForwardPropogation(vector<double>& x){

     // Calculate the Forget Gate 
     vector<double> f;  

     for (int i=0; i<num_inputs; i++){         

         double sum = 0.0;          

         for (int j=0; j<num_inputs; j++)    sum += Wf[i][j] * x[j];            // Weighted Sum of inputs to the gate         

         sum += bf[i];                                                          // Add the bias         

         f.push_back(1/(1 + exp(-sum)));                                        // Apply sigmoid activation function 
     }       

    // Calculate the Input Gate 
    vector<double> i;  

    for (int i=0; i<num_inputs; i++){         

        double sum = 0.0;          

        for (int j=0; j<num_inputs; j++)    sum += Wi[i][j] * x[j];            // Weighted Sum of inputs to the gate         

        sum += bi[i];                                                         // Add the bias         

        i.push_back(1/(1 + exp(-sum)));                                       // Apply sigmoid activation function 
    }       

    // Calculate the Output Gate 
    vector<double> o;  

    for (int i=0; i<num_inputs; i++){         

        double sum = 0.0;          

        for (int j=0   j<num_inputs; j++)    sum += Wo[i][j] * x[j];            // Weighted Sum of inputs to the gate
    // Add the bias         

        sum += bo[i];                                                         // Add the bias         

        o.push_back(1/(1 + exp(-sum)));                                       // Apply sigmoid activation function 
    }           
    // Calculate the Cell State
    vector<double> c;           // Cell State                       
    vector<double> h;           // Hidden State

    for (int i=0; i<num_inputs; i++){         

        double sum = 0.0;          

        for (int j=0; j<num_inputs; j++)    sum += Wc[i][j] * x[j];            // Weighted Sum of inputs to the gate         

        sum += bc[i];                                                         // Add the bias         

        c.push_back(tanh(sum));                                               // Apply tanh activation function 
    }

    // Calculate the Hidden State
    for (int i=0; i<num_inputs; i++)    h.push_back(o[i] * c[i]);               // Apply the output gate to the cell state

    // Print the results
    cout << "Forget Gate: " << endl;
    for (int i=0; i<num_inputs; i++)    cout << f[i] << " ";
    cout << endl;

    cout << "Input Gate: " << endl;
    for (int i=0; i<num_inputs; i++)    cout << i[i] << " ";
    cout << endl;

    cout << "Output Gate: " << endl;
    for (int i=0; i<num_inputs; i++)    cout << o[i] << " ";
    cout << endl;

    cout << "Cell State: " << endl;
    for (int i=0; i<num_inputs; i++)    cout << c[i] << " ";
    cout << endl;

    cout << "Hidden State: " << endl;
    for (int i=0; i<num_inputs; i++)    cout << h[i] << " ";
    cout << endl;

    }   

    void LSTM::BackPropogation(vector<vector<double>>& delta){              // Backpropogation algorithm



    }

    void LSTM::UpdateWeights(){                     // Update the weights and biases of the model
                                  

    }                   
    int main(){         

        // Create an LSTM object 
        LSTM lstm(3);         

        // Initialize the weights and biases 
        lstm.InitWeights();         

        // Create an input vector 
        vector<double> x;         

        x.push_back(0.1);         
        x.push_back(0.2);         
        x.push_back(0.3);         

        // Perform forward propogation 
        lstm.ForwardPropogation(x);         

        return 0;     
    }                                       

    




    


    
    
