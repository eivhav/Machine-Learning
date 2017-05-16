package oving2;

import java.util.Arrays;
import java.util.List;

public class Rule {
	
	
	private ElementCount antecedent;
	private ElementCount consequent; 
    private double support;
    private double confidence;
    List<String[]> transactions; 
	
	public Rule(ElementCount antecedent, ElementCount consequent, List<String[]> transactions){
		this.transactions = transactions; 
		this.antecedent = antecedent; 
		this.setConsequent(consequent); 
		
	}
	
	
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
    	
    public double calculateConfidence(ElementCount resultElemennt){
    	double x = (double) supportCount(antecedent); 
    	double xUy = resultElemennt.getCount();  
    	confidence = xUy / x; 
    	return confidence;   
    }
    
    public void printRule(){
    	String anc = antecedent.getElements()[0];  
    	String cons = getConsequent().getElements()[0]; 
    	for(int i = 1 ; i < antecedent.getElements().length; i++){
    		anc = anc + " " + antecedent.getElements()[i]; 
    	}
    	for(int i = 1 ; i < getConsequent().getElements().length; i++){	
    		cons = cons + " " + getConsequent().getElements()[i];
    	}
    	
    	System.out.print(anc + " --> " + cons + "  " + confidence + "\n"); 
    }


	public ElementCount getConsequent() {
		return consequent;
	}


	public void setConsequent(ElementCount consequent) {
		this.consequent = consequent;
	}

	public ElementCount getAntecedent() {
		return antecedent;
	}


	public void setAntecedent(ElementCount antecedent) {
		this.antecedent = antecedent;
	}


	public double getSupport() {
		return support;
	}


	public void setSupport(double support) {
		this.support = support;
	}


	public double getConfidence() {
		return confidence;
	}


	public void setConfidence(double confidence) {
		this.confidence = confidence;
	}
}
