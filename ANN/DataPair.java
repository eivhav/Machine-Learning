package Exc5;

import java.util.ArrayList;

public class DataPair {
	
	public ArrayList<Float> highFeatures;  
	public ArrayList<Float> lowFeatures;  
	
	public DataPair(DataInstance first, DataInstance second){
		if(first.rating > second.rating){
			this.highFeatures = first.features;
			this.lowFeatures = second.features; 
		}
		else{
			this.highFeatures = second.features;
			this.lowFeatures = first.features; 
		}	
		
	}
	

}
