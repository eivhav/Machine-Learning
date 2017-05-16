package oving2;

//import no.ntnu.idi.tdt4300.arg.apriori.*;

import java.util.ArrayList;
import java.util.List;

import java.io.BufferedReader;

import java.io.InputStreamReader;

/* Much of the skeleton code has been unused and some has been rewritten due to problems with the ItemSet and its superclasses. 
 * We choice instead to use Arrays and ArrayList.
 *  
 */

public class AprioriAssociationRuleGenerator {
	
	public static final String SMALL_DATASET = "/data/smallDataset.txt";
    public static final String LARGE_DATASET = "/data/supermarket.arff";
    
 
	
	
    public static void main1(String[] args) {
        System.out.println("Starting (dataset=" + args[0] + ", minsup=" + args[1] + ", minconf=" +  args[2] + ")...");

        List<String[]> transactions = null;
        switch (args[0]) {
		    case "small":
		        transactions = getDataFromClass("small");
		        break;
		    case "large":
		        transactions = getDataFromClass("large");
		        break;
		    default:
		        throw new IllegalArgumentException();
		}

        // support and confidence threshold
        Double minSupport = Double.parseDouble(args[1]);
        Double minConfidence = Double.parseDouble(args[2]);

        // generating the association rules
        Fkm1Fkm1Algorithm aprioriAlgorithm = new Fkm1Fkm1Algorithm(transactions);

        aprioriAlgorithm.generate(minSupport, minConfidence, -1); 
   
        System.out.println("Finished :-)");
    }
   
	
	
	// Extracts the data from the data class instead of txt-files.
	public static List<String[]> getDataFromClass(String size){ 
		ArrayList<String[]> returnList = new ArrayList<String[]>(); 
		Data data = new Data(); 
		
		if(size.equals("small")){
			for(int i = 0; i < data.getSmallDataList().size(); i++){
				ArrayList<String> transaction = new ArrayList<String>(); 
				String dataLine = data.getSmallDataList().get(i); 
				for(int j = 0; j < dataLine.length(); j++){
					if(dataLine.charAt(j) == 't'){
						transaction.add(data.getSmallAtrList().get(j)); 
					}
				}
				String[] trans = ListToArrayConverter(transaction); 	
				returnList.add(trans); 
			}
		}
		else if(size.equals("large")){
			for(int i = 0; i < data.getLargeDataList().size(); i++){
				ArrayList<String> transaction = new ArrayList<String>(); 
				String dataLine = data.getLargeDataList().get(i); 
				for(int j = 0; j < dataLine.length(); j++){
					if(dataLine.charAt(j) == 't'){
						transaction.add(data.getLargeAtrList().get(j)); 
					}
				}
				String[] trans = ListToArrayConverter(transaction); 	
				returnList.add(trans); 
				}
		}	
		return returnList; 
		
	}
	
  
	
	  // Secondary main method for console input. 
	public static void main(String[] args) { 
	    	List<String[]> transactions = getDataFromClass("large");
	    	Double minSupport = 0.40; 
	    	Double minConfidence = 0.7;  
	    	int supportCount = -1; 
	    	
	    	Fkm1Fkm1Algorithm aprioriAlgorithm = new Fkm1Fkm1Algorithm(transactions);

	        aprioriAlgorithm.generate(minSupport, minConfidence, supportCount);
	        
	        System.out.println("Finished :-)");
	        
	    }
	
	
	
    
   // this method enables input by the console, where each transaction represented as a line and each attributed separated by comma.  
    // the output is a list(transactions) with String arrays which contains the attributes. 
	public static List<String[]> textReaderInn(String[] args){
		ArrayList<String[]> transactions = new ArrayList<String[]>();     
    	BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
         
         
         try {
             String linje = in.readLine();
             while (linje != null) {
            	 linje.trim(); 
            	 ArrayList<String> set = new ArrayList<String>(); 
            	 int digitPos= 0; 
            	 String element = "";
            	 while(linje.length()!= 0){
            		 if(linje.charAt(digitPos) == ','){
            			 set.add(element); 
            			 element= new String(); 
            		 }
            		 else{ 
            			element = element + linje.charAt(digitPos); 
            		 }
            	 	if(linje.length() > 1)
            	 		linje = linje.substring(digitPos +1); 
            	 	else 
            	 		linje = ""; 
            	 }
            	 
            	 String[] elementArray = new String[set.size()];  
            	 for(int i = 0 ; i < set.size(); i++){
            		 elementArray[i] = set.get(i); 
            	 }
            	 transactions.add(elementArray); 
            	 linje = in.readLine();
             }   
         }
         
         
         catch (Exception e) {
             e.printStackTrace();
         }
    	
         return transactions; 
         
    }
	
	
  
   
	//Helper methods, checks for duplicates
    public static boolean checkForDuplicates(ArrayList<String> list, String item){
    	boolean tempReturn = false; 
    	for(int i = 0 ; i < list.size(); i++){
    		if(item.equals(list.get(i))){
    			tempReturn = true; 
    			break; 
    		}
    	}
    	return tempReturn; 
    }
    
    
    //Helper methods, ArrayList--> array converter. 
    public static String[] ListToArrayConverter(ArrayList<String> list){
    	String[] returnString = new String[list.size()]; 
    	for(int i = 0 ; i < list.size(); i++){
    		returnString[i] = list.get(i); 
    	}
    	return returnString; 	
    }
    
    
    


}
