package Exc5;

import java.util.ArrayList;

public class DataInstance{
	
	public int queryId; 
	public double rating;  
	public ArrayList<Float> features = new ArrayList<Float>(); 
	
	public DataInstance(int queryId, double rating, ArrayList<Float> features){
        this.queryId = queryId;  			//ID of the query
        this.rating = rating; 				//Rating of this site for this query
        this.features = features; 			//The features of this query-site pair.
}
        
        

    public String printDataInstance(){
    	return "Datainstance - qid: "+ queryId+ ". rating: "+ rating + ". features: "+ printArrayListAsLine(features); 
    	
    }
        
    public String printArrayListAsLine(ArrayList<Float> list){
    	String out = ""; 
    	for(int i = 0; i < list.size(); i++){
    		out += list.get(i) + " "; 
    	}
    	return out; 
    	
    	
    }

}