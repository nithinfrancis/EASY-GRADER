function validate(){
           var f = myForm.elements["first_name"].value;
           var l = myForm.elements["last_name"].value;
           var e = myForm.elements["email"].value;
           var cp = myForm.elements["cp"].value;
           var up = myForm.elements["up"].value;
           var c = myForm.elements["contact_no"].value;
           if(f==""||e==""||up==""|cp==""||c==""||l==""){
               alert("complete all fields");
               return false;
           }
           else if (cp!=up) {
             alert("password doesnot match")
             return false;
           }
       }
