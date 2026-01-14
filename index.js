import 'dotenv/config'
import {Agent, run} from "@openai/agents"
const helloAgent = new Agent({
    name: 'hello agent',
    instructions: "you are an agent that always says hello-world",
});

run(helloAgent, "hey my name is mubashir rao").then((result)=>{
    console.log(result.finalOutput);
})
