import java.util.*

Class TestCheck{
  int id;
  static int num;
  
 public TestCheck(int val){
  id = val;
  num++;
 }
 
 public static void main(String str[]){
    TestCheck aTestCheck = new TestCheck();
    SYstem.out.println(aTestCheck.num);
 }
}
