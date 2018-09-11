
//Existing function
//Might be useful
function getCurrentTabUrl(callback) {
  var queryInfo = {
    active: true,
    currentWindow: true
  };

  //Some permissions thing I really dont care about
  chrome.tabs.query(queryInfo, (tabs) => {
    var tab = tabs[0];
    var url = tab.url;
    console.assert(typeof url == 'string', 'tab.url should be a string');
    callback(url);
  });

}


function checkTerms(color) {
  var script = 'document.body.style.backgroundColor="' + color + '";';
  // See https://developer.chrome.com/extensions/tabs#method-executeScript.
  // chrome.tabs.executeScript allows us to programmatically inject JavaScript
  // into a page. Since we omit the optional first argument "tabId", the script
  // is inserted into the active tab of the current window, which serves as the
  // default.
  var someVar = {text: 'test', foo: 1, bar: false};
  chrome.tabs.executeScript({
    code: '(' + function(params) {
//      alert(document.body.innerText);
      console.log(document.body.innerText);

      //****************************************************/

      //To get all the hyperlinks
      // var getLinks = function(){
      //   linked_data = "";
      //   links = document.links;
      //   link_array = [];
      //   for (var i=0;i<links.length;i++){
      //     if (links[i].innerHTML.toLowerCase().includes("privacy") 
      //         || links[i].innerHTML.toLowerCase().includes("terms")   
      //           || links[i].innerHTML.toLowerCase().includes("policy")){
      //             link_array.push(links[i].href)
      //           };
            
      //   };
        
      //   for (count in link_array){
      //     //document.getElementById(link_array[count]).click();

      //     var popup = window.open(link_array[count],"_self");

      //     // $(popup).load(function() {
      //     //   linked_data+=document.body.innerText+"\n";
      //     //   history.back();
      //     //   );

      //     $(popup.document).load(function() {
      //       linked_data+=document.body.innerText+"\n";
      //       history.back();
      //       // do other things
      //   });

        
      //   };

      //   return linked_data;
      // };

      var x = 5;

      //The blessed function for sending requests
      function post(path, params, method) {
        method = method || "post"; // Set method to post by default if not specified.
    
        // The rest of this code assumes you are not using a library.
        // It can be made less wordy if you use one.
        var form = document.createElement("form");
        form.setAttribute("method", method);
        form.setAttribute("action", path);
    
        for(var key in params) {
            if(params.hasOwnProperty(key)) {
                var hiddenField = document.createElement("input");
                hiddenField.setAttribute("type", "hidden");
                hiddenField.setAttribute("name", key);
                hiddenField.setAttribute("value", params[key]);
    
                form.appendChild(hiddenField);
             }
        }
    
        document.body.appendChild(form);
        form.submit();
    }
    //****************************************************************** */

    var data = document.body.innerText;

    data = data.split("\t").join(" ");
    data = data.split("\n").join(" ");
    data = data.split("\r").join(" ");

    post('http://127.0.0.1:3000/parseHere/', {pageData: data});
    
        //var data = document.body.innerHTML+" "+ getLinks()
        
        //data.replace("\\n"," ").replace("\\t"," ").replace("\\r"," ");

        

        return {success: true, data};
    } + ')(' + JSON.stringify(someVar) + ');'
}, function(results) {
    console.log(results[0]);
});

}


//When the page loads
document.addEventListener('DOMContentLoaded', () => {
  getCurrentTabUrl((url) => {
    var dropdown = document.getElementById('dropdown');
    var el = document.body;
    var text = el.innerText;

    //When they pick Check
    dropdown.addEventListener('change', () => {
      alert(el.textContent);
      checkTerms(dropdown.value);
      
    });
  });
});



