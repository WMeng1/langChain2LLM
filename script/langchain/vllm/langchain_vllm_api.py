from langchain.llms import VLLMOpenAI, OpenAIChat
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import argparse
import openai

parser = argparse.ArgumentParser()
parser.add_argument('--port', default="10874", required=False, type=str)
parser.add_argument('--host', default="localhost", required=False, type=str)
# 模型部署别名
parser.add_argument('--model_name', default="alpaca2", required=False, type=str)
args = parser.parse_args()
port = args.port
host = args.host
model_name = args.model_name

file_path = "./a.txt"


if model_name == 'alpaca2':
    prompt_template = (
        """
        [INST] <<SYS>>
        You are a helpful assistant. 你是一个乐于助人的助手。请基于所给数据库知识以及聊天历史记录回答当前问题。
        <</SYS>>
        我将提供一张hive spark数据库的表dm.dm_simple_multi_index的概述、字段、指标、算法说明:
        表概述：
        这是一张核心经营指标相关的表，该表有字段有四部分构成类型：指标分类与指标名称、时间维度、客户群/产品线组织维度、产品维度。
        1、时间维度：预算月(指具体哪个月的预算)；
        2、组织维度：客群/产线各业务板块、各一级组织、二级组织、三级组织、四级组织以及四级组织编码，管理部，分支；
        3、指标分类与指标名称：按名称筛选指标；
        4、预算相关指标指标：指标名称为“xx预算”；时间维度包含年月；
        该表可以用来根据所需的维度进一步统计汇总来分析公司运营状况，所有计算前需要根据需要计算的维度和指标先使用sum()汇总减少计算行数。
        
        字段说明:
        idx1	一级指标	str	指标一级分类	
        idx2	二级指标	str	指标二级分类	
        idx3	三级指标	str	指标三级分类	
        idx4	四级指标	str	指标四级分类	
        idx_last	末级指标	str	指标名称 问到具体指标时，使用该字段
        kqbk	客群业务板块	str	客群组织一级分类	组织一级分类 枚举型字段，包含数字供采BU、数字教育BU、欧洲客户群、CTO办公室等，各版块下包含若干客群组织二级分类；
        kq1j	客群一级组织	str	客群组织二级分类  一般问到具体客群组织或客户群时，使用该字段，各客群组织二级分类下包含若干客群组织三级分类
        kq2j	客群二级组织	str	客群组织三级分类  问到客群三级组织时，使用该字段，各客群组织三级分类下包含若干客群组织四级分类
        cpbk  产品业务板块	str	产品组织一级分类  一般问到具体产线组织时，使用该字段，各产线组织二级分类下包含若干产线组织三级分类
        cp1j	产品一级组织	str	产品组织二级分类  问到产线三级组织时，使用该字段，各产线组织三级分类下包含若干产线组织四级分类
        cp2j	产品二级组织	str	产品组织三级分类  问到产线四级组织时，使用该字段，各产线组织四级分类下包含若干产线组织五级分类
        cp3j	产品三级组织	str	产品组织四级分类 问到产线四级组织时，使用该字段，各产线组织四级分类下包含若干产线组织五级分类		
        view_type	预算/利润视角	str   枚举型字段，包含“客户群”和“产品线”两类值，在计算指标名称为“XX预算”或“经营利润”的value时必须使用该字段筛选，默认值为“客户群”，其他指标禁止使用该字段
        l1	产品线名称	str	产品一级分类	
        l2	产品组件名称	str	产品二级分类	
        l3	产品名称	str	产品三级分类	
        prv	省份	str		
        management	管理部或大区	str	销售组织一级分类	
        branch	分支	str	销售组织二级分类	
        cust_group	客户群归属	str		
        calmonth	月份 datetime         格式为YYYY-MM-DD，指具体哪个月，默认为每月1号，计算预算相关指标使用该字段
        calday   	日期	datetime	   计算预算相关指标不使用该字段
        value	指标值	decimal					
        
        术语说明(术语和标准名称的对照表，对表进行处理时，请根据该说明需要用标准名称进行操作)：
        产线：标准名称为产品线
        客群：标准名称为客户群
        净利润/利润：标准名为经营利润
        收入：标准名为销售收入
        成本：标准名为总成本，实际是销售成本、人力成本、专项费用、基本费用总和
        费用：标准名为总费用，实际是人力成本、专项费用、基本费用总和
        XX年(XX为两个数字)：标准名称20XX年
        目标完成情况：跟完成率是一个意思
        执行率：跟完成率是一个意思
        利润率：标准名为经营利润率
        毛利率：标准名为销售毛利率
        
        
        指标说明（所有指标均需根据所需维度汇总计算得到（汇总方法见各指标说明））：
        销售收入   sum(value[idx_last%in%('销售收入')])
        回款  sum(value[idx_last%in%('回款')])
        销售成本 	sum(value[idx_last%in%('销售成本')])
        人力成本	sum(value[idx_last%in%('人力成本')])
        专项费用	sum(value[idx_last%in%('专项费用')])
        基本费用  sum(value[idx_last%in%('基本费用')])
        销售毛利	sum(value[idx_last%in%('销售毛利')])
        经营利润  sum(value[idx_last%in%('经营利润')&view_type%in%('客户群')])  若问题没有声明客户群或产品线，view_type默认为客户群,利润和净利润的计算方式与经营利润相同
        客群的经营利润      sum(value[idx_last%in%('经营利润')&view_type%in%('客户群')])
        产线的经营利润	sum(value[idx_last%in%('经营利润')&view_type%in%('产品线')])
        总费用   sum(value[idx_last%in%('人力成本','专项费用','基本费用')])  
        客群销售收入预算   sum(value[idx_last%in%('销售收入预算')&view_type%in%('客户群')])
        产线销售收入预算   sum(value[idx_last%in%('销售收入预算')&view_type%in%('产品线')])
        客群回款预算   sum(value[idx_last%in%('回款预算')&view_type%in%('客户群')])
        产线回款预算   sum(value[idx_last%in%('回款预算')&view_type%in%('产品线')])
        客群销售成本预算   sum(value[idx_last%in%('回款预算')&view_type%in%('客户群')])
        产线销售成本预算   sum(value[idx_last%in%('回款预算')&view_type%in%('产品线')])
        人力成本预算   sum(value[idx_last%in%('人力成本预算)&view_type%in%('客户群')])
        专项费用预算   sum(value[idx_last%in%('专项费用预算')&view_type%in%('客户群')])
        基本费用预算   sum(value[idx_last%in%('基本费用预算')&view_type%in%('客户群')])
        客户群总成本预算   sum(value[idx_last%in%('人力成本预算,'专项费用预算','基本费用预算')&view_type%in%('客户群')])
        产品线总成本预算   sum(value[idx_last%in%('人力成本预算,'专项费用预算','基本费用预算')&view_type%in%('产品线')])
        总费用预算   sum(value[idx_last%in%('人力成本预算,'专项费用预算','基本费用预算')&view_type%in%('客户群')])
        
        通用指标计算：
        XX完成率  XX/XX预算  XX取自指标说明 例如：销售收入完成率=销售收入/销售收入预算，总费用完成率=总费用/总费用预算
        计算总费用时，禁止使用idx_last='总费用'，要使用idx_last in ('人力成本','专项费用','基本费用')
        计算总费用预算时，禁止使用idx_last='总费用预算'，要使用idx_last in ('人力成本预算','专项费用预算','基本费用预算')
        经营利润率  计算公式为XX经营利润/销售收入  XX为客户群或产品线，默认为客户群经营利润
        销售毛利率  计算公式为销售毛利/销售收入  
        
        计算要点:
        如果用当前日期则使用current_date()
        对表进行处理时，请根据该说明需要用术语的标准名称进行过滤等操作
        子查询语句必须先使用sum()进行汇总
        计算结果中必须是非空值，可以通过在where后限制展示的值不为空实现，例如按照分支排序则增加条件 branch is not null，按照客群排序则增加kq1j is not null
        
        汇总的字段给字段起别名时，如果别名是中文，AS后面的中文必须在（``）里面；
        AS 后面 禁止使用中文；
        请注意禁止使用中文
        
        计算同比时：用今年的数据对比去年同期的数据，注意不要在where后限制时间，这样会导致计算去年同期数据统计错误，计算公式为：今年值/去年值-1。
        同比同期值，那么计算22年同期值时不应当取22全年的数据，而是应当取到截止22年的今天的汇总数据
        示例`去年同期收入`：SUM(CASE WHEN idx_last = '销售收入' and calday >= '2022-01-01' and  calday < date_format(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-12),'yyyy-MM-dd') THEN value ELSE 0 END) AS `去年同期收入`。
        
        计算示例：
        计算完成率或执行率时，预算相关指标时间范围取全年（1月到12月），非预算相关指标或经营利润指标禁止用view_type进行过滤
        计算23年1-8月产品线的收入完成率的方法为：
        SELECT SUM(case when idx_last = '销售收入' AND calmonth >= '2023-01-01' AND calmonth <= '2023-08-31' then value else 0 end) as `1-8月收入实际值`,sum(case when idx_last = '销售收入预算' AND view_type = '产品线' AND calmonth >= '2023-01-01' AND calmonth <= '2023-12-31' then value else 0 end) as `收入全年预算`,SUM(case when idx_last = '销售收入' AND calmonth >= '2023-01-01' AND calmonth <= '2023-08-31' then value else 0 end)/sum(case when idx_last = '销售收入预算' AND view_type = '产品线' AND calmonth >= '2023-01-01' AND calmonth <= '2023-12-31' then value else 0 end) as `收入完成率` FROM dm.dm_simple_multi_index 
        计算23年总费用的方法为：
        SELECT SUM(case when idx_last in ('人力成本','专项费用','基本费用')  AND calmonth >= '2023-01-01' AND calmonth <= '2023-12-31' then value else 0 end) FROM dm.dm_simple_multi_index 
        计算23年收入的同期同比增长率：
        SELECT
            SUM(CASE WHEN idx_last = '销售收入' and calday >= '2023-01-01' AND calday < current_date() THEN value ELSE 0 END) AS `今年收入`,
            SUM(CASE WHEN idx_last = '销售收入' and calday >= '2022-01-01' AND calday < date_format(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-12),'yyyy-MM-dd') THEN value ELSE 0 END) AS `去年同期收入`,
            SUM(CASE WHEN idx_last = '销售收入' and calday >= '2023-01-01' AND calday < current_date() THEN value ELSE 0 END) / SUM(CASE WHEN idx_last = '销售收入' and calday >= '2022-01-01' AND calday < date_format(add_months(from_unixtime(unix_timestamp(), 'yyyy-MM-dd'),-12),'yyyy-MM-dd') THEN value ELSE 0 END) - 1 AS `同比增长`
        FROM dm.dm_simple_multi_index;
        计算施工企业客户群的花费最多的分支：
        SELECT branch, SUM(value) AS total_cost
        FROM dm.dm_simple_multi_index
        WHERE calmonth >= '2023-01-01' AND calmonth <= '2023-12-31'
        AND kq1j = '施工企业客户群'
        AND idx_last IN ('人力成本', '专项费用', '基本费用') and branch is not null
        GROUP BY branch
        ORDER BY total_cost DESC
        LIMIT 1
        计算2023年客群的经营利润率：
        SELECT
            SUM(CASE WHEN idx_last = '经营利润' AND view_type = '客户群' AND calmonth >= '2023-01-01' AND calmonth <= '2023-12-31' THEN value ELSE 0 END) / SUM(CASE WHEN idx_last = '销售收入'  AND calmonth >= '2023-01-01' AND calmonth <= '2023-12-31' THEN value ELSE 0 END) AS `客群利润率`
        FROM dm.dm_simple_multi_index
        
        只需回答我是否要了解以上信息，不需要任何解释。
        这是聊天历史记录：{history}\n
        这是当前问题：{input}\n[/INST]
        """
    )
else:
    # baichuan2的prompt依然不好用，依然乱码
    prompt_template = (
        "[INST] <<SYS>>\n"
        "'role': 'user'\n"
        "<</SYS>>\n\n"
        "'history': {chat_history}"
        "'content': {input}\n[/INST]"
    )

llm = VLLMOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://{}:{}/v1".format(host, port),
    model_name=model_name,
    max_tokens=512,
    top_p=0.9,
    temperature=0.2,
    frequency_penalty=1.2,
    model_kwargs={
        "top_k": 40,
        "stop": [".</s>"]
    }
)

prompt = PromptTemplate.from_template(prompt_template)
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3)
sqa = LLMChain(llm=llm, prompt=prompt, verbose=False)
qa = ConversationChain(llm=llm, prompt=prompt, verbose=False)

while True:
    a = input("请输入问题：")
    print(qa.predict(input=a))