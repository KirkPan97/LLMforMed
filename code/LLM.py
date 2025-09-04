from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

def LLMforMed(model_name, data_path):
    model_path = f"/home/panky/PROJ/LLM/Models/Qwen/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    print("Model loaded successfully.")

    raw_data = pd.read_excel(data_path)
    data = raw_data.sort_values(by='ID1')

    result_df = pd.DataFrame(columns=['name', 'Characteristics', 'response'])

    for i in tqdm(range(len(raw_data))):

        name = data.iloc[i]['姓名']
        age = data.iloc[i]['初次诊断年龄']
        height = data.iloc[i]['身高（cm）']
        weight = data.iloc[i]['体重（kg）']
        hypertension = data.iloc[i]['高血压（0：无；1:轻度；2:中度；3:重度）']
        if hypertension == 0:
            hypertension = 'no'
        elif hypertension == 1:
            hypertension = 'mild'
        elif hypertension == 2:
            hypertension = 'moderate'
        elif hypertension == 3:
            hypertension = 'severe'
        diabetes = data.iloc[i]['糖尿病（0:无；1:有）']
        if diabetes == 0:
            diabetes = 'no'
        elif diabetes == 1:
            diabetes = 'yes'
        er = data.iloc[i]['新辅助治疗前ER（0阴性，1阳性，2未做）']
        if er == 0:
            er = 'negative'
        elif er == 1:
            er = 'positive'
        elif er == 2:
            er = 'unavailable'
        pr = data.iloc[i]['新辅助治疗前PR（0阴性，1阳性，2未做）']
        if pr == 0:
            pr = 'negative'
        elif pr == 1:
            pr = 'positive'
        elif pr == 2:
            pr = 'unavailable'
        ki67 = data.iloc[i]['新辅助治疗前Ki67( )具体值']
        her2 = data.iloc[i]['新辅助治疗前HER2（0无扩增，1有扩增，2临界值不确定;3未做）']
        if her2 == 0:
            her2 = 'no'
        elif her2 == 1:
            her2 = 'yes'
        elif her2 == 2:
            her2 = 'uncertain'
        elif her2 == 3:
            her2 = 'unavailable'
        t_stage = data.iloc[i]['新辅助治疗前T分期']
        n_stage = data.iloc[i]['新辅助治疗前N分期']
        m_stage = data.iloc[i]['新辅助治疗前M分期']
        histo_type = data.iloc[i]['新辅助治疗前病理组织学类型（1.乳腺浸润性癌，非特殊类型；2浸润性导管癌；3导管原位癌，4浸润性小叶癌，5小叶原位癌，6.浸润性导管癌伴导管内癌 7粘液癌8神经内分泌癌9印戒细胞癌10鳞癌）']
        if histo_type == 1:
            histo_type = 'invasive breast cancer'
        elif histo_type == 2:
            histo_type = 'invasive ductal carcinoma'
        elif histo_type == 3:
            histo_type = 'ductal carcinoma in situ'
        elif histo_type == 4:
            histo_type = 'invasive lobular carcinoma'
        elif histo_type == 5:
            histo_type = 'lobular carcinoma in situ'
        elif histo_type == 6:
            histo_type = 'invasive ductal carcinoma with intraductal carcinoma'
        elif histo_type == 7:
            histo_type = 'mucinous carcinoma'
        elif histo_type == 8:
            histo_type = 'neuroendocrine carcinoma'
        elif histo_type == 9:
            histo_type = 'signet-ring cell carcinoma'
        elif histo_type == 10:
            histo_type = 'squamous carcinoma'
        stage = data.iloc[i]['新辅助治疗前分期（1:IA;2:IB;3:IIA;4:IIB;5IIIA;6IIIB;7IIIC;8IV）']
        if stage == 1:
            stage = 'IA'
        elif stage == 2:
            stage = 'IB'
        elif stage == 3:
            stage = 'IIA'
        elif stage == 4:
            stage = 'IIB'
        elif stage == 5:
            stage = 'IIIA'
        elif stage == 6:
            stage = 'IIIB'
        elif stage == 7:
            stage = 'IIIC'
        elif stage == 8:
            stage = 'IV'
        response = data.iloc[i]['新辅助后肿瘤退缩分级（0完全缓解；1非完全缓解）']
        if response == 0:
            response = 'pCR'
        elif response == 1:
            response = 'non-pCR'

        Characteristics = f"Characteristics: "
        Characteristics += f"age: {age}, "
        Characteristics += f"height: {height}cm, "
        Characteristics += f"weight: {weight}kg, "
        Characteristics += f"hypertension: {hypertension}, "
        Characteristics += f"diabetes: {diabetes}, "
        Characteristics += f"ER: {er}, "
        Characteristics += f"PR: {pr}, "
        Characteristics += f"ki-67: {ki67}, "
        Characteristics += f"HER2: {her2}, "
        Characteristics += f"T stage: {t_stage}, "
        Characteristics += f"N stage: {n_stage}, "
        Characteristics += f"M stage: {m_stage}, "
        Characteristics += f"stage: {stage}, "
        Characteristics += f"histo type: {histo_type}. "

        prompt = f"Predict pathological complete response after neoadjuvant chemotherapy (NAC) of a breast cancer patient based on her clinical characteristics before NAC. All you need to tell me in one word (pCR/non-pCR) is whether this patient is most likely to reach pCR or non-pCR. "
        prompt += Characteristics

    
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        new_row = pd.DataFrame([{
            'name': name,
            'Characteristics': Characteristics,
            'response': response,
            'thinking_content': thinking_content,
            'predict': content
        }])
        result_df = pd.concat([result_df, new_row], ignore_index=True)

        with open(f"/home/panky/PROJ/LLMforMed/result/txt/{model_name}/{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"Characteristics: {Characteristics}\nresponse: {response}\nthinking_content: {thinking_content}\npredict: {content}")

        print(f"name: {name} is done. ")

    output_path = f"/home/panky/PROJ/LLMforMed/result/excel/{model_name}_result.xlsx"
    result_df.to_excel(output_path, index=False, engine="openpyxl")

LLMforMed("Qwen3-4B-Thinking-2507", "/home/panky/PROJ/BCUS/data/乳腺癌NAC队列20211028.xlsx")

