package oving2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class Fkm1Fkm1Algorithm {
	 
	public ArrayList<ElementCount> allItems = new ArrayList<ElementCount>(); 
	public ArrayList<ElementCount> resultItems = new ArrayList<ElementCount>(); 
	public static ArrayList<ArrayList<String[]>> ruleCanditatesStrings = new ArrayList<ArrayList<String[]>>(); 
	public static ArrayList<Rule> finalRules = new ArrayList<Rule>();
	public double minSupport; 
	public double minConfidence; 
	protected List<String[]> transactions;
    
    
	public Fkm1Fkm1Algorithm(List<String[]> transactions){
		this.transactions = transactions; 
	}
	
 
    public void generate(double minSupport, double minConfidence, int supportCount) { // the main algorithm. 
    	
    	if(supportCount == -1){
    		double sCount = transactions.size() * minSupport; 
    		supportCount = roundUp(sCount); 
    	}
    	singleItemsGenerator(); 
    	generateSupportCount(allItems);
    	System.out.print("Generated elements and the supportcount" +"\n"); 
    	printElements(allItems); 
    	
    	System.out.print("Pruning" +"\n");
    	pruning(supportCount, allItems);
    	printElements(allItems); 
    	System.out.print("-----------------" + "\n" + "\n");
    	
    	boolean keepJoining = true; 
    	//ArrayList<ElementCount> copy = new ArrayList<ElementCount>(); 
    	
    	while(keepJoining ){ 
    		//copy= copyArray(allItems); 
    		ArrayList<ElementCount> newList = join(allItems);
    		if(newList.size() != 0){
    			allItems = newList; 
    			generateSupportCount(allItems); 
    			System.out.print("Generated new elements and the supportcount" +"\n"); 
    			printElements(allItems); 
    			System.out.print("Pruning" +"\n");
    			pruning(supportCount, allItems); 
    			printElements(allItems);
    			for(int i = 0; i < allItems.size(); i++){
    				resultItems.add(allItems.get(i));
    			}
    		}
    		else{ 
    			keepJoining = false;
    			System.out.print("Final elements" +"\n");
    			printElements(resultItems);
    		}
    		
    		if(allItems.size() == 0){
    			//allItems= copy; 
    			keepJoining = false;
    			System.out.print("All elements pruned away!" + "\n" + "\n" + "Final elements:" +"\n");
    			printElements(resultItems);  
    		}
    		System.out.print("-----------------" + "\n" + "\n");
    	
    	}
   
    			
    	this.minConfidence = minConfidence;  
    	if(resultItems.size() > 0){
    		if(resultItems.get(0).getElements().length > 1){
    		generateRuleCanditates(resultItems); 
        	generateFinalRules(); 
        	printRules();
    		}
    		else{
    		System.out.print("No rules to be made with the current minSup" +"\n");
    		}
    	}
    }
    
    
    
    // The itemset generation and pruning part of the code 
    
    // Takes the transaction table  and extracts all the attributes(items). Lists them in the allItems list only once each, alphabetical.  	
    public void singleItemsGenerator(){
    	for(int i = 0 ; i < transactions.size(); i++){
    		String[] itemSet = transactions.get(i);
    		for(int j = 0; j < itemSet.length; j++){
    			String item = itemSet[j]; 	
    			boolean inList = false; 
    			for(int k = 0 ; k < allItems.size(); k++){
    				if(item.equals(allItems.get(k).getElements()[0])){
    					inList = true; 
    					break; 
    				}
    			}
    			if(!inList){
    				String[] element = new String[1];  
    				element[0] = item; 
    				ElementCount elementCount = new ElementCount(element, 0);  
    				elementCount.setElements(element);
    				int pos= -1; 
    				for(int k = 0 ; k < allItems.size(); k++){
    					if(elementCount.getElements()[0].compareToIgnoreCase(allItems.get(k).getElements()[0]) < 0){
    						pos = k; 	
    						break; 
    				}
    				}
    				if(pos == -1){
    					allItems.add(elementCount);}
    				else{
    					allItems.add(pos,elementCount); 
    				}
    			}		  				
    		}		
    	}
    }
  
    
    // Used for printing the items and their count. 
    public void printElements(ArrayList<ElementCount> elements){
    	for(int i = 0 ; i < elements.size(); i++){ 
    		ElementCount element = elements.get(i); 
    		for(int j = 0 ; j < element.getElements().length; j++){  
    			System.out.print(element.getElements()[j] + " ");  
    		}
    		System.out.print(" " + element.getCount() + "\n" ); 
    	}
    	System.out.print("\n");
    }
    
    
    
    // Finds items to join by traversing down the allItems list.   
    public ArrayList<ElementCount> join(ArrayList<ElementCount> elementList){
    	ArrayList<ElementCount> result = new ArrayList<ElementCount>(); 
    	if(elementList.size() > 0){
	    	int round = elementList.get(0).getElements().length -1; 
	    	for(int i = 0 ; i < elementList.size(); i++){
	    		for(int j = i+1 ; j < elementList.size(); j++){  
	    			ArrayList<String> stringElement = joinResult(elementList.get(i), elementList.get(j), round); 
	    			if(stringElement != null){	
	    				String[] newArray = new String[stringElement.size()]; 
	    				newArray= stringElement.toArray(newArray); 
	    				result.add(new ElementCount(newArray, 0)); 
	    			}
	    		}
	    	}
    	}
    	return result; 	
    }
  
    
    // Takes inn to and to items, and checks if they can be joined. If so, it return the  result of the join. Else null.   
    public ArrayList<String> joinResult(ElementCount elem1, ElementCount elem2, int round){
    	String[] elements1 = elem1.getElements(); 
    	String[] elements2 = elem2.getElements(); 
    	String[] elem1New = Arrays.copyOfRange(elements1, 0, elements1.length -1);  
		String[] elem2New = Arrays.copyOfRange(elements2, 0, elements2.length -1); 

    	ArrayList<String> joinElements = new ArrayList<String>(); 
    	boolean joinable = false;   
    	
    	if(Arrays.equals(elem2New, elem1New)){ 
    		for(int i = 0 ; i < elem1New.length; i++){ 
    			joinElements.add(elem1New[i]); 
    		}
    		joinElements.add(elements1[elements1.length-1]);
    		joinElements.add(elements2[elements2.length-1]);
    		joinable= true;	
    	} 			 
    	if(joinable){
    		return joinElements; } 
   		else{
   			return null; }
    }
    	
    
    // Takes inn a set of elements(itemSets) and checks if the count is below the support count. If so, it removes them from the list. 
    public void pruning(int supportCount, ArrayList<ElementCount> elements){
    	for(int i = 0 ; i < elements.size(); i++){
    		if(elements.get(i).getCount() < supportCount){
    			elements.remove(i); 
    			i--; 
    		}
    	}
    }		
   
    
    // Sets the suport count value for each element(itemset) 
    public void generateSupportCount(ArrayList<ElementCount> elementList){
    	for(int i = 0 ; i < elementList.size(); i++){
    		elementList.get(i).setCount(supportCount(elementList.get(i))); 
    	}			
   	}
    
    
    // Calculates the support count for a specific itemSet by looking at the transaction list.  
    public int supportCount(ElementCount element){
    	int count = 0; 
   		String[] itemsForSearch = element.getElements(); 
   		for(int i = 0 ; i < transactions.size(); i++){
   			String[] trans = transactions.get(i); 
    		boolean isIntransaction = true; 
    		for(int j = 0 ; j < itemsForSearch.length; j++){
    			if(!Arrays.asList(trans).contains(itemsForSearch[j])){
    				isIntransaction = false; 
    			}
   			}
   			if(isIntransaction){  		
   				count++; 
    			}
    		}
    		return count; 
    }
    
    
    
    // uses the next method for finding all substrings of the freqentItemset and adds them to the ruleCandidatesStrings list. Does so for each itemset of the result elements  
    public static void generateRuleCanditates(ArrayList<ElementCount> resultItems){
    	for(int i = 0; i < resultItems.size(); i++){
    		
    		ArrayList<String[]> ruleList = new ArrayList<String[]>(); 
    		if(resultItems.get(i).getElements().length == 2){ 
    			String[] el1 = new String[1]; 
    			el1[0] = resultItems.get(i).getElements()[0];  
    			String[] el2 = new String[1]; 
    			el2[0] = resultItems.get(i).getElements()[1];  
    			ruleList.add(el1); 
    			ruleList.add(el2); 
    			ruleCanditatesStrings.add(ruleList); 
    		}
    		else{
    			generateRuleStringList(resultItems.get(i).getElements(), ruleList); 
        		ruleCanditatesStrings.add(ruleList); 
        		
    		}
    		
    		}
    }
     
    	
    
    	
    	// Finds subsets of an itemSet with length of n/2 + 1.  
	    public static void generateRuleStringList(String freqItems[], ArrayList<String[]> list){
			for(int i = 1; i < freqItems.length -1; i++){
				 String[] data = new String[i]; 
				 combGenerator(freqItems, data, 0, (freqItems.length-1), 0, i, list);
			 }	
		}
	    	// Recursive method for finding all permutations av an element set. 
			public static  void combGenerator(String arrg[], String dataInput[], int start, int end, int index, int x, ArrayList<String[]> returnList){
			     if (index == x){
			    	 String[] element = new String[x]; 
			         for (int j = 0; j < x; j++){
			             element[j] = dataInput[j]; 
			         }
			         returnList.add(element); 
			         
			         return;
			     }
			     
			     for (int i = start; i <= end && end - i + 1 >= x - index; i++){
			    	if(index < end/2 +1){
			    		dataInput[index] = arrg[i];
			    		combGenerator(arrg, dataInput, i+1, end, index+1, x, returnList); 
			    	}
			     }
			 }
	
	
			
	// Starts the process of creating and adding rules to the finalRules list. 		
	 public void generateFinalRules(){
	    	for(int i = 0; i < ruleCanditatesStrings.size(); i++){
	    		ElementCount resultItem = resultItems.get(i);  
	    		createRules(ruleCanditatesStrings.get(i), resultItem); 
	    	}		
	    }
	    
	 	//Creates a new rule based on the element permutations at rulesCandiateStrings.
	 	//First it iterates down the list, and then back up with the consequent/antecedent order reversed.   
	    public void createRules(ArrayList<String[]> rulesCandStrings, ElementCount resultItem){
	    	for(int i = 0; i < rulesCandStrings.size(); i++){ 
	    		createAndEstimateNewRule(rulesCandStrings, resultItem, i); 
	    	}
	    	int middleSize = rulesCandStrings.get(rulesCandStrings.size()-1).length; 
	    	int totalSize = resultItem.getElements().length; 
	    	for(int i = rulesCandStrings.size() -1; i >=0; i--){
	    		if(middleSize == totalSize/2 && rulesCandStrings.get(i).length <= middleSize -1){
	    			createAndEstimateNewRuleReverese(rulesCandStrings, resultItem, i);  
	    		}
	    		else if(middleSize != totalSize/2 && rulesCandStrings.get(i).length < middleSize -1) {
	    			createAndEstimateNewRuleReverese(rulesCandStrings, resultItem, i); 
	    		}
	    		else if(middleSize == 1){
	    			createAndEstimateNewRuleReverese(rulesCandStrings, resultItem, i);
	    		}
	    	}	
	    }
	    
	    	// Creates the rule, with antecedent and consequent. Calculates the confidence, and if > minConf, it adds the rule to the finalRule list.   
		    public void createAndEstimateNewRule(ArrayList<String[]> rulesCandStrings, ElementCount resultItem, int i){
		    	ElementCount ancendent = new ElementCount(rulesCandStrings.get(i), 0);  
				ElementCount conseq = new ElementCount(resultItem.getOtherSubset(ancendent.getElements()), 0); 
				Rule rule = new Rule(ancendent, conseq, transactions);  
				if(rule.calculateConfidence(resultItem) >= minConfidence){
					finalRules.add(rule);
				}
		    }
		    
		    // Does the same as above, but with antecedent and consequent switched
		    public void createAndEstimateNewRuleReverese(ArrayList<String[]> rulesCandStrings, ElementCount resultItem, int i){
		    	ElementCount ancendent = new ElementCount(rulesCandStrings.get(i), 0);  
				ElementCount conseq = new ElementCount(resultItem.getOtherSubset(ancendent.getElements()), 0); 
				Rule rule = new Rule(conseq, ancendent, transactions);  
				if(rule.calculateConfidence(resultItem) >= minConfidence){
					finalRules.add(rule);
				}
		    }
	
		  
		    
	// Prints all rules in the finalRules list.  
    public void printRules(){	 
    	System.out.print("Rules: " + "\n"); 
    	for(int i = 0; i < finalRules.size(); i++){
    		finalRules.get(i).printRule(); 
		   }
	 }
		    
    // Help method for copying an arrayList. 
     public ArrayList<ElementCount> copyArray(ArrayList<ElementCount> orgList){
    	  ArrayList<ElementCount> returnArray = new ArrayList<ElementCount>();  
    	  for(int i = 0 ; i < orgList.size(); i++){
    		  returnArray.add(orgList.get(i)); 
    	  }
    	  return returnArray; 		
      }
    	  
      
     
     public int roundUp(double num){
    	 int intNum = (int) num;  
    	 double doubNum = (double) intNum; 
    	 if(num == doubNum){
    		 return intNum; 
    	 }
    	 else {
    		 return intNum+1; }	 
     }
      
      						
   
      	
    	
   }
    
