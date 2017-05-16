package oving2;

public class ElementCount {
	
	// Contains a set of elements and the associated support count. 
	
	private String[] elements; 
	private int count;
	
	public ElementCount(String[] elements, int count){ 
		this.elements = elements; 
		this.count = count; 
	}
	
	public String[] getElements() {
		return elements;
	}
	public void setElements(String[] elements) {
		this.elements = elements;
	}
	public int getCount() {
		return count;
	}
	public void setCount(int count) {
		this.count = count;
	} 
	
	public String getPrintElements(){
		String out = "";  
		for(int i = 0; i < elements.length; i++){
			out = out + elements[i] + " "; 
		}
		return out; 
	}
	
	public String[] getOtherSubset(String[] subset){
		String[] otherSubset = new String[elements.length - subset.length];
		if(elements.length > 2){
			int count = 0; 
			for(int i = 0; i < elements.length; i++){
				String elem = elements[i]; 
				boolean isInsubset = false; 
				for(int j = 0; j < subset.length; j++){
					if(elem.equals(subset[j])){
						isInsubset = true; 
						break; 
					}
				}
				if(isInsubset== false){
					otherSubset[count] = elem; 
					count++; 
				}
			}
		}
		else if(elements.length == 2){
			if(elements[0] != subset[0]){
				otherSubset[0] = elements[0]; 
			}
			else {
				otherSubset[0] = elements[1]; 
			}
				
		}
			
			
		return otherSubset; 
	}

}
