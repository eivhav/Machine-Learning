package Exc5;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Scanner;
import java.util.StringTokenizer;

public class dataLoader {
	
	public static ArrayList<ArrayList<DataInstance>> testData = new ArrayList<ArrayList<DataInstance>>(); 
	public static ArrayList<ArrayList<DataInstance>> trainingData = new ArrayList<ArrayList<DataInstance>>(); 
	
	
	//The main / start of the program.  
	public static void main(String[] args){
		System.out.println("Loading training data. Please wait..."); 
		loadData(trainingData, "C:\\train.txt"); 
		System.out.println("Training data loaded. Number of Queries: " + trainingData.size()); 
		System.out.println("Loading test data. Please wait..."); 
		loadData(testData, "C:\\test.txt"); 
		System.out.println("Test data loaded. Number of Queries: " + testData.size()); 
		System.out.println();
	
		for(int i = 0; i < 5; i++){
			System.out.println("Round " + (i+1)); 
			ArrayList<ArrayList<Float>> results = runRanker(); 
			printResults(results); 
			System.out.println(); 
			System.out.println("----------------------------"); 
		}
	
	}
	
	
	//Creates patterns of all the dataPairs, creates a new NN object and counts misorderdPairs for the testPattern.  
	//Then for 25 epochs, it trains the net, tests with the testing pattern and reports.   
	public static ArrayList<ArrayList<Float>> runRanker(){
		System.out.println("Running ranker"); 
		ArrayList<ArrayList<Float>> results = new ArrayList<ArrayList<Float>>(); 
		results.add(new ArrayList<Float>()); 			//trainingResults
		results.add(new ArrayList<Float>()); 			//testResults
 
    	ArrayList<ArrayList<DataPair>> trainingPatterns = addToPatterns(trainingData); 
    	ArrayList<ArrayList<DataPair>> testPatterns = addToPatterns(testData);
    	
    	NN nn = new NN(46,10, 0.001); 
    	System.out.println("Cehcking ANN before traning: "); 
    	results.get(1).add((float) nn.countMisorderedPairs(testPatterns));  	//Check ANN performance before training
    	
    	for(int i = 0; i < 150; i++){
    		results.get(0).add((float) nn.train(trainingPatterns,1)); 
    		results.get(1).add((float)nn.countMisorderedPairs(testPatterns)); 	
    	}
    	return results; 
	}
	

	
	
	
	//Creates the complete testPattern, by running createDataPairs for all Queries.  
	public static ArrayList<ArrayList<DataPair>> addToPatterns(ArrayList<ArrayList<DataInstance>> instanceLists){
		System.out.println("Creating patterns"); 
		ArrayList<ArrayList<DataPair>> patterns = new ArrayList<ArrayList<DataPair>>();  
		for(int i = 0; i < instanceLists.size(); i++){
			patterns.add(createDataPairs(instanceLists.get(i))); 
		}
		return patterns; 
	}
	
	
	//Creates dataPairs for features with same qId. 
	public static ArrayList<DataPair> createDataPairs(ArrayList<DataInstance> instanceList){
		ArrayList<DataPair> returnList = new ArrayList<DataPair>();  
	
		for(int i = 0; i < instanceList.size() -1; i++){
			DataInstance primal = instanceList.get(i); 
			for(int j = i+1; j < instanceList.size(); j++){
				DataInstance secondary = instanceList.get(j); 
				if(primal.rating != secondary.rating){
					returnList.add(new DataPair(primal, secondary)); 
					
				}
			}
		}
		return returnList; 
	}
	
	
	
	
//-------- The method that read and loads the data ----------//

	//Takes in either the testData or the training data holding list, and extracts data from filepath.  
	//Each feature/line creates a DataInstance object, which then is put in a list with others with same queryId.   
	//Each query-list is thereafter stored in the complete ArrayList testData/traingData. 
	public static void loadData(ArrayList<ArrayList<DataInstance>> input, String filepath){
			File file = new File((filepath)); 
			FileInputStream fis = null; 
			BufferedInputStream bis = null; 
			DataInputStream dis = null; 
			
			try { 
				fis = new FileInputStream(file);
				bis = new BufferedInputStream(fis); 
				dis = new DataInputStream(bis); 
				

				while (dis.available() != 0) {  
					String line =  dis.readLine(); 
					ArrayList<Float> features = new ArrayList<Float>(); 
					int rating = 0; 
					int qId = 0; 
					Scanner sc = new Scanner(line);
					rating = sc.nextInt();
					line = line.substring(6); 
					sc = new Scanner(line);
					qId = sc.nextInt(); 
					line = line.substring(5);
					
					for(int i = 0; i < 46; i++){
						if(i < 9){
							line = line.substring(3);}
						else {
							line = line.substring(4);} 
						sc = new Scanner(line); 
						sc.useLocale(Locale.US); 
						float feature = sc.nextFloat(); 
						features.add(feature); 
						line = line.substring(8);
					}
					DataInstance dataInst = new DataInstance(qId, rating, features); 
					CheckAndPlaceQid(qId, input, dataInst); 	
				}
				fis.close(); 
				bis.close(); 
				dis.close(); 
			}
			catch (FileNotFoundException e) { 
			e.printStackTrace(); 
			}
			catch (IOException e) { 
			e.printStackTrace(); 
			} 
	} 
	
	
	//Places the DataInstance in the list that contains other with the same Query ID. 
	//If such list exist, it creates a new list for its qid.  
	public static void CheckAndPlaceQid(int Qid, ArrayList<ArrayList<DataInstance>> list, DataInstance di){
		boolean inList = false; 
		int pos = -1; 
		for(int i = 0; i < list.size(); i++){
			if(list.get(i).get(0).queryId == Qid){
				inList = true; 
				pos = i; 
				break; 
			}
		}
		if(inList){
			list.get(pos).add(di); 
		}
		else {
			ArrayList<DataInstance> qidlist = new ArrayList<DataInstance>(); 
			qidlist.add(di); 
			list.add(qidlist); 
		}	
	}
		

	
	
//------ The printing method ----------- //
	
	public static void printResults(ArrayList<ArrayList<Float>> results){
		System.out.println("Printing results:" + "\n"); 
		System.out.println("Training error:"); 
		for(int i = 0; i < results.get(0).size(); i++){
			System.out.println(results.get(0).get(i));
		}
		System.out.println();
		System.out.println("Testing error:"); 
		System.out.println(results.get(1).get(0) + ": init"); 
		for(int i = 1; i < results.get(1).size(); i++){
			System.out.println(results.get(1).get(i));
		}
	}
	
	
	
	
}
