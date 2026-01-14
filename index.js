import 'dotenv/config'
import {Agent, run} from "@openai/agents"
const location = 'pakistan'
const helloAgent = new Agent({
    name: 'hello agent',
    instructions: function (){
        if (location === 'pakistan'){
            return "Hello from Pakistan!";
        }else if (location === 'india'){
            return "Hello from India!";
        }else{
            return "Hello from the rest of the world!";
        }
    }
});

run(helloAgent, "hey my name is mubashir rao").then((result)=>{
    console.log(result.finalOutput);
})
