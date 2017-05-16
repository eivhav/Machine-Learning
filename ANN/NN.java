package Exc5;

import java.util.ArrayList;

public class NN {
	
	public int numInputs; 
	public int numHidden; 
	public double learningRate = 0.001; 
	
	public ArrayList<Float> inputActivations; 
	public ArrayList<Float> hiddenActivations; 
	public float outputActivation; 
	public ArrayList<Float> prevInputActivations; 
	public ArrayList<Float> prevHiddenActivations; 
	public float prevOutputActivation; 
	
	public float deltaOutput = 0; 
	public ArrayList<Float> deltaHidden; 
	public float prevDeltaOutput = 0; 
	public ArrayList<Float> prevDeltaHidden;
	
	public float weightsInput[][]; 
	public float weightsOutput[]; 
	
	
	//Creates a new NN- network, with no of inputs, num hidden nodes, and learning rate.  
	public NN(int numInputs, int numHidden, double LR){
		this.numInputs = numInputs+1; 									// +1 for bias node: A node with a constant input of 1. Used to shift the transfer function.
		this.numHidden = numHidden; 
		this.learningRate = LR; 
		// Current activation levels for nodes (in other words, the nodes' output value)
		inputActivations = createInitActivationLIst(this.numInputs, 1.0); 
		hiddenActivations = createInitActivationLIst(this.numHidden, 1.0); 
		outputActivation = (float) 1.0; //Assuming a single output.
		//create weights
		weightsInput = new float [this.numInputs][this.numHidden]; 		//A matrix with all weights from input layer to hidden layer
		weightsOutput = new float [this.numHidden]; 					//A list with all weights from hidden layer to the single output neuron.
		// set them to random values
		for(int i = 0; i < numInputs; i++){
			for(int j = 0; j < numHidden; j++){
				weightsInput[i][j] = randomFloat(-0.5, 0.5); 
			}
		}	
		for(int i = 0; i < numHidden; i++){
			weightsOutput[i] = randomFloat(-0.5, 0.5); 
		}
		//Data for the backpropagation step in RankNets. For storing the previous activation levels (output levels) of all neurons
		prevInputActivations = new ArrayList<Float>(); 
		prevHiddenActivations = new ArrayList<Float>(); 
		prevOutputActivation = (float) 0; 
		deltaHidden = createInitActivationLIst(this.numHidden, 0);		//For storing the current delta in the same layers
		prevDeltaHidden = createInitActivationLIst(this.numHidden, 0); 	//For storing the previous delta in the hidden layer
		System.out.println("Creating ANN with learning rate: " + LR); 
	}
	
	
	
    //Trains the the network. Propagates A(the highest rated features), then B and then backPropagtes to update the weights.  
    public double train(ArrayList<ArrayList<DataPair>> pattern, int iterations){
    	for(int i = 0; i < pattern.size(); i++){
    		ArrayList<DataPair>  dataset = pattern.get(i); 
    		for(int j = 0; j < dataset.size(); j++){
    			DataPair datapair = dataset.get(j); 
    			propagate(datapair.highFeatures); 
    			propagate(datapair.lowFeatures); 
    			backpropagate(); 
    		}
    	}
    	return countMisorderedPairs(pattern); 
    }
        
   
    
    //Returns the rate of correctly rated pairs. Return the error-rate.  
    public double countMisorderedPairs(ArrayList<ArrayList<DataPair>> patterns){
    	double numRight = 0; 
    	double numMisses = 0;
    	double errorRate = -1; 
    	for(int i = 0; i < patterns.size(); i++){
    		ArrayList<DataPair>  dataset = patterns.get(i); 
    		for(int j = 0; j < dataset.size(); j++){
    			DataPair datapair = dataset.get(j); 
    			propagate(datapair.highFeatures); 
    			propagate(datapair.lowFeatures); 
    			
    			if(prevOutputActivation > outputActivation){
    				numRight++; 
    			}
    			else if(prevOutputActivation < outputActivation){
    				numMisses++; 
    			}
    		}
    	}
    	if(numRight + numMisses > 0){
    		 errorRate = numMisses / (numRight + numMisses); 
    	}
    	return errorRate; 
    }
	
	
	
    //Sends a feature list thru the network and updates the activations values. 
	public void propagate(ArrayList<Float> features){
		if(features.size() != numInputs -1){
			System.out.println("wrong number of inputs"); 
			return; }
		else {
			//input activations
			prevInputActivations = copyArray(inputActivations); 
			for(int i = 0; i < numInputs -1; i++){
				inputActivations.set(i, features.get(i));  
			}
			inputActivations.set(numInputs -1, (float) 1);  //Set bias node to -1. ???
		
			//hidden activations
			prevHiddenActivations = copyArray(hiddenActivations); 
			for(int j = 0; j < numHidden ; j++){
				float sum = (float) 0.0; 
				for(int i = 0; i < numInputs; i++){
					//print ai[i] ," * " , wi[i][j]
					sum = sum + (inputActivations.get(i) * weightsInput[i][j]); 
				}
				hiddenActivations.set(j, logFunc(sum)); 
			}
			
			//output activations
			prevOutputActivation = outputActivation; 
			float sum = (float) 0.0;
			for(int j = 0; j < numHidden ; j++){
				sum = sum + hiddenActivations.get(j) * weightsOutput[j]; 
			}
			outputActivation = logFunc(sum); 
		}
	}


	
    //Backpropagets, computes the deltas, and updates the wheights. 
    public void backpropagate(){
        computeOutputDelta(); 
        computeHiddenDelta(); 
        updateWeights(); 
    }
	
	
    
	//Implements the delta function for the output layer (see exercise text)
    public void computeOutputDelta(){ 
    	float outputAdelta = (logFuncDerivative(prevOutputActivation) * (1- calculatePab(prevOutputActivation, outputActivation)));    
    	float outputBdelta = (logFuncDerivative(outputActivation) * (1- calculatePab(prevOutputActivation, outputActivation)));  
    	prevDeltaOutput = outputAdelta; 
    	deltaOutput = outputBdelta;
    }

    
    //Implements the delta function for the hidden layer (see exercise text) 
    public void computeHiddenDelta(){ 
    	for(int i = 0; i < deltaHidden.size(); i++){  
    		float hiddenAdelta = logFuncDerivative(prevHiddenActivations.get(i)) * weightsOutput[i] * (prevDeltaOutput-deltaOutput);  
    		float hiddenBdelta = logFuncDerivative(hiddenActivations.get(i)) * weightsOutput[i] * (prevDeltaOutput-deltaOutput); 
    		prevDeltaHidden.set(i, hiddenAdelta); 
    		deltaHidden.set(i, hiddenBdelta);	
    	}
    }

    
    //Updates the weights of the network using the deltas (see exercise text)
    public void updateWeights(){ 
    	for(int j = 0; j < numHidden ; j++){ 
			for(int i = 0; i < numInputs; i++){
				weightsInput[i][j] = (float) (weightsInput[i][j] + (learningRate *((prevDeltaHidden.get(j)*prevInputActivations.get(i)) - (deltaHidden.get(j)*inputActivations.get(i)))));   
			}
			weightsOutput[j] = (float) (weightsOutput[j] + (learningRate *((prevDeltaOutput*prevHiddenActivations.get(j)) - (deltaOutput*hiddenActivations.get(j))))); 	
    	}
    }

    
    //------- Assistance methods ------------
    
    //Copies he float from one ArrayList and creates a new one.(Deep copy equivalent) 
	public ArrayList<Float> copyArray(ArrayList<Float> initArrayList){
		ArrayList<Float> returnList = new ArrayList<Float>(); 
		for(int i = 0; i < initArrayList.size(); i++){
			returnList.add(initArrayList.get(i)); 
		}
		return returnList; 
	}
	

	//Creates and fills a list of length size and inits it with all entities as value. 
	public ArrayList<Float> createInitActivationLIst(int size, double value){
		ArrayList<Float> returnList = new ArrayList<Float>(); 
		for(int i = 0; i < size; i++){
			returnList.add((float) value); 
		}
		return returnList; 
	}
	
	
	//Returns a random float value between d and e. 
	public float randomFloat(double d, double e){
		return  (float) (Math.random() * (e-d) + d);
	}
	
	
	// Returns a log value, used by updating nodes  
	private Float logFunc(float x) {
		return (float) (1.0/(1.0 + (float) Math.exp(-x))); 
		
	}
	
	
	// Returns a derivative of log value, used by updating weights 
	private float logFuncDerivative(float x){
	    return (float) Math.exp(-x)/((float) Math.pow(Math.exp(-x)+1, 2));   
	}
	
	
	//Calculates the Pab value from output a and b. 
	public Float calculatePab(float outA, float outB){ 
		return (float) (1.0/(1.0 + (float) Math.exp(outB - outA)));  
	}
    
    

}
