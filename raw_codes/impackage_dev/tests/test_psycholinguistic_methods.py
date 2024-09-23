# Databricks notebook source
# MAGIC %run "./../algorithms/psycholinguistic_methods"

# COMMAND ----------

import numpy as np

# COMMAND ----------

"""
Nutter Fixture for testing the config module.
"""

from runtime.nutterfixture import NutterFixture
class PsycholinguisticMethodsFixture(NutterFixture):
   """
   This Statistics fixture is used for unit testing all the methods that are used in the statistics.py module 
   """
   def __init__(self):
      """
      helps in intializing all the instance variables

      """
      self.psycholinguistic_obj=PsycholinguisticMethods()
      self.txt=[
  "As a summary upfront, in a very challenging environment, Austrian Post has had a good start into the year with strong revenue growth and despite significant headwinds, I think we can look confident into the full year 2023.",
  "Page 2 reminds you of the three segments that we operate in and that we report.",
  "Mail, our incumbent Austrian mail business; Parcel & Logistics, our portfolio of parcel networks in nine geographies in Austria, Southeastern Europe and Türkiye; and other segment Retail and Bank, including our post office network and our bank99.",
  "Moving to page 3, we are operating in a challenging macroeconomic environment with three strong drivers impacting our business.",
  "First we see a weak sentiment, a consolidation in e-commerce, but also a weak environment in stationery, among stationery retailers; second, we see inflation across our cost structure, not only affecting staff cost, energy costs, but pretty much all elements of our cost structure; the third particular driver, given that we have a quite strong exposure to Türkiye, Türkiye still shows high inflation, at the same time, the Turkish lira exchange rate to the euro has been surprisingly stable, which has led to strong growth numbers coming from Türkiye in Euro.",
  "But at the same time, this also means there is a risk once the Turkish lira depreciates and purchasing power – purchasing powers adjust again.",
  "Page 4 gives you an overview of the first of the highlights of Q1.",
  "I'll talk about the numbers later on.",
  "I think the summary is double-digit growth both top line as well as bottom line.",
  "Significant challenges are addressed by multiple efforts, both on the cost side as well as on price management.",
  "And overall, we do confirm the outlook, we've given two months ago with revenue target – with a target of revenue increase in the mid-single digit range, was depending on the Turkish lira exchange rate.",
  "Still a strong CapEx year with CapEx spending in the order of magnitude of €160 million to €180 million and an earnings target where the objective is to get close to last year's earnings level.",
  "Page 5 gives you a multi-year representation of Q1 revenue split.",
  "I think there are two messages.",
  "Message number one is we do have a strong, profitable and resilient mail business.",
  "And I think message two is, over the years, we have built up a strong second leg of our Parcel and Logistics business, which is the growth driver and which, after a year of consolidation is back in growth mode again, with revenue growth of 15.5%.",
  "I think third message there is a financial service business, which is also growing again after concluding the operation as part of Moving to page 6, our revenue has – we have seen a quarter with strong revenue growth, as already said, 10.5%.",
  "Roughly half of that is coming from Türkiye, where the already mentioned combination of strong inflation with a relatively stable Turkish lira has led to a very strong growth in Europe, almost 70% – 66%, roughly.",
  "Still without Türkiye, I think a growth of 5.5% is a very good number given the structurally declining Mail business and a relatively weak sentiment both in e-commerce as well as in stationery retail.",
  "Our Mail business, has grown revenues by 3.3%.",
  "Price changes have overcompensated the decline in volumes.",
  "Parcel and Logistics, plus 15.5%, roughly 5%, excluding Türkiye and the Retail and Bank segment growing around €10 million or 42%.",
  "Page 7, gives you the overview of earnings last year of 39.6%, this year plus 18.7% to €47 million with our core business Mail and Parcel in absolute terms pretty stable, which given growing revenues means there is some visible margin pressure as we are not as – as it is challenging to forward all the cost increases that we see across our cost structure to the market.",
  "Still, I think overall decent margins in both businesses.",
  "Retail and Bank is a very strong improvement of €12 million, the first break-even quarter after three years.",
  "Please do not expect this to be in a breakeven situation for the next three quarters as significant migration expenses will weigh on our results in this segment over the next quarters.",
  "But I think still, it gives us a good sign, a very encouraging sign that our bank99 is on a good way.",
  "Corporate's down a little bit from last year and overall, again, €47 million.",
  "Page 8 reminds you of our strategy.",
  "Number one is defending our market leadership and profitability in our core business, which means, our core Mail and Parcel businesses.",
  "Priority number two, profitable growth in near markets, meaning international growth, but also growth along the value chain and priority number three is to further develop offerings for consumers and SMB.",
  "Here bank99 is one of the biggest measures and the green arrow in the middle, we do aim to be a leader in sustainable logistics and diversity and customer focus complemented strategy element.",
  "Let me now along the strategy framework update you on the developments in our core business lines, starting with addressed letter mail.",
  "Addressed letter mail has shown to be very resilient element of our business throughout the pandemic.",
  "The biggest acceleration of digitization in our society which happened as a result of the pandemic, has not let mail volumes to decline significantly faster than in the past.",
  "Also, Q1 2023 was a mail decline on a daily basis of minus 4%, shows continuous – continued stable, relatively moderate decline.",
  "We have raised prices significantly, moving to page 10 last year, in three steps.",
  "The Eco product in July, the Priority product in October and pretty much all other products in January this year.",
  "With these price changes, we remain one of the cheapest mail Page 11.",
  "Direct Mail remains a strong element of our Mail business portfolio.",
  "Direct Mail and Media Post last year totaling €427 million in revenues.",
  "This has been the business which has suffered significantly during the pandemic and has not – and probably will not recover fully from the pandemic as we have seen a strong shift towards e-commerce, away from the stationery retailers and stationery retailers are the core customer base of this business.",
  "Strong pressure continues to weigh on this product and on our customers in We see an increasing number of industry exits and insolvencies in that field.",
  "But still we are working hard every day to keep this advertising, to keep this important element of the marketing mix of our customers relevant and to invest also in digital – digital marketing as a way to extend the traditionally paper-based Direct Mail products.",
  "Moving to page 12 or to our Parcel business in Austria.",
  "Last year has been a year of consolidation with mail volumes declining slightly but modestly.",
  "This year we are back in the growth mode as already in the second half of last year, plus 5% volumes in Q1 2023.",
  "And also in particular in the last months, we have seen a good momentum in our parcel volumes.",
  "We continue with our investment program.",
  "2023 will be pretty much the last year of a major expansion program in our Austrian logistics infrastructure, the last big project we are currently working on is an expansion of our historically most important sorting center in the south of Vienna.",
  "This is meant to go live or go into operations in the second half of this year.",
  "Together with substantial fleet investments, IT investments and investments in the businesses outside the Austrian business, our investments this year, our CapEx spending will total roughly €160 million to €180 million.",
  "Q1 that's pretty much every year has been a relatively weak start into the CapEx year.",
  "Page 14 shows you the investment program that first started in 2018, you see here that is logistic center in Page 15 shows you the development of our staff structure.",
  "I think messages here is the transformation towards the New Collective Wage Agreement being the dominant element in our workforce and civil servant, and the Old Collective Labor Agreement employees continuing to decline.",
  "A second message is we have improved efficiency over the last 12 months.",
  "Despite growing parcel volumes, we have managed to get through Q1 with almost 600 FTEs less than last year.",
  "I think the results of the efficiency improvement measures and more stable operations and also a result of the investments we have done over the last years.",
  "Moving to our International Parcel business, again here good growth, again after a consolidation last year, volume growth in Eastern Europe, plus 12%, in Türkiye plus 8%, price improvements in CEE a little bit more challenging in Türkiye in the high inflation.",
  "Of course, a lot of price changes you see on page 17 some more details about our Turkish business.",
  "Our cargo is again growing substantially also on a real base, if you try to find a measure for that but also taking into account that there is substantial inflation, we do see as already said, parcel volumes growing at high single-digit numbers and the company operates on a good profitability.",
  "Moving to our third strategic pillar, our Retail business is here the strongest investment, of course, has been our investment into bank99, as I said, the first pretty much breakeven quarter after launching this bank three years ago.",
  "In particular, interest income has increased substantially, given the changes – the steep changes in the interest rate environment.",
  "Interest income on a quarterly, for Q1, increasing from 7 million last year to almost 16 million this year.",
  "We see this increase over the next quarters in Q4.",
  "Already last year we have already included some significant interest rate changes.",
  "With that, as I said, a breakeven quarter.",
  "Over the next roughly five to six quarters, we expect significant integration and migration expenses as we now have made the decision to how we will integrate the two core banking systems and this migration project is starting out these days.",
  "The other big priority these days is to make sure we remain attractive for customer deposits by introducing parcel savings products, of course, without paying too much interest on them.",
  "This gives you an update about our self-service solution we continue to invest a lot in self-service solutions.",
  "It's not really big money that we spend here, but it's – we think this is an important element of – an important source of differentiation in an increasingly competitive parcel market.",
  "These solutions are well accepted by our customers.",
  "Talk a little bit about our progress on the sustainability front.",
  "Two important priorities in our sustainability programs are e-mobility and photovoltaics with 3,000 vehicles end of last year.",
  "We operate by far the largest fleet – electric fleet in Austria.",
  "We do have a target to eliminate the last combustion engine in our last mile fleet by 2030.",
  "End of this year, we plan to have more than 3,800 vehicles.",
  "Next year, more than 4,600 vehicles.",
  "So this is well on track.",
  "The other priority where we have accelerated our investments and activities is photovoltaics.",
  "Given the price changes in the electricity market, we have substantially accelerated our PV expansion program.",
  "Last year, added roughly 1.5 megawatt peak.",
  "This year, we will double, almost double the installed capacity on a little bit about 4 or around 8 megawatt peak.",
  "And so next year, we do have a pipeline that should lead us to 15 megawatt peak, 15 megawatt peak, is the capacity which should generate roughly 20% of the electricity that's needed, including the electricity for our electric fleet from renewable sources.",
  "Our various sustainability efforts I think are well appreciated also by different rating agencies where we typically come out best in class or among the top players in our industry.",
  "Several awards that are shown on page 23, both on our efforts as well as on our reporting also demonstrate the recognition of our efforts.",
  "Let me now progress with some more details on our financials.",
  "Page 24 shows you a few financial KPIs.",
  "I have already commented on revenues.",
  "For margins EBITDA as well as EBIT above last year.",
  "I think we are all aware that Q1 was a difficult one.",
  "So it is partly also a recovery earnings per share and cash flow, I think on a decent level for Q1, operating free cash flow of €75 million was that order of magnitude.",
  "I think we're in a good way of earnings in free cash flow from operations to be in a good position to propose another attractive dividend for the current year at the end of the year.",
  "Page 25 shows you our group P&L. I don't want to go into the details of that, let me rather proceed with going into the individual segments on page 26.",
  "Starting with Mail Distribution, the core Letter Mail business including our Business Solutions activities showed revenue increase of €5.6 million.",
  "As I mentioned, a relatively stable decline in volumes of around 4%, plus significant price changes.",
  "Next to this revenue increase in the Direct Mail side, the volume losses, and this is a combination of unaddressed and addressed direct mail losses, were again compensated by price changes.",
  "So, the revenue impact was, in total, relatively small, minus 0.8%.",
  "With those revenues and very disciplined cost focus approaching our operations, we were able to secure decent margins in Mail, with a total EBIT of €41 million, and an EBIT margin of around 13%.",
  "Page 28 moving to Parcel and Logistics here.",
  "Revenues up 15.5% with all regions contributing to this growth.",
  "Austria, Türkiye and Eastern Europe.",
  "I already mentioned that Turkish lira, inflation In Logistics Solutions, we show a decline in revenues.",
  "This is coming from one-off revenues from the pandemic that Page 29 shows you the segment P&L. The – pretty much stable.",
  "EBIT in absolute terms, the loss of special pandemic-related logistics services here, overcompensating the increase in absolute EBIT generated by the other activities.",
  "And as a result, the small EBIT compression from 65%.",
  "Page 30, moving to Retail & Bank, here revenue growth of 41.8% predominantly coming from the top line growth in bank99, plus almost 60%, roughly €10 million in top line growth in the bank.",
  "And as I said on page 31 you can see that here we can show the first full quarter with the segment breakeven of two years we hope to see more such quarters in the future.",
  "Page 32 updates you on the structure of our balance sheet.",
  "We continue to operate a healthy and conservative balance sheet with low level of financial debts with those banks of €150 million on loan, with conservative accounting, which is demonstrated by high level of provisions, in particular staff-related provisions, low level of intangible assets and goodwill and quarter-by-quarter increasing balance sheet, which is driven by the growth of bank99.",
  "Page 33 update is the usual illustration of our cash flows.",
  "We try to focus on the column in the middle.",
  "Operating free cash flow before growth CapEx.",
  "This is where we have the target to generate an operating free cash flow that well covers the dividend with €75 million in the first quarter.",
  "I think we're in a good way.",
  "Let me finish with the outlook on page 34, which is pretty much confirming the outlook we gave two months ago.",
  "We continue to see a challenging market environment with strong inflation and weak consumer sentiment.",
  "In this environment we target a mid-single digit revenue growth with the growth again coming from Parcel & Logistics, Mail for the full year probably, a slight revenue decline and on the Retail & Bank segment, we will continue to see revenue increase driven by the improved interest rate environment.",
  "On the CapEx side we confirmed a €100-million – roughly €100 million maintenance CapEx, plus €60 million to €80 million growth CapEx.",
  "And on the earnings side, despite significant inflationary challenges leading to increases staff cost and With that said, I'm ready to take your questions.",
  "So, here with me is our CFO, Walter Oblin and I would like to directly hand over to Walter."
]
      NutterFixture.__init__(self)

   def assertion_LM_analysis_per_section(self):
      """
      This method is used for unit testing StatisticsFixture.LM_analysis_per_section method.
      """
      
      val=self.psycholinguistic_obj.LM_analysis_per_section(self.txt)
      assert (val[4]==0 and val[5]== 12 and val[6]==20)

   def assertion_LM_analysis_empty_text(self):
      """
      This method is used for unit testing StatisticsFixture.LM_analysis method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.LM_analysis_per_section(empty_txt)
      assert (np.isnan(val[4]) and np.isnan(val[5]) and np.isnan(val[6]))

   def assertion_LM_analysis_per_sentence(self):
      """
      This method is used for unit testing StatisticsFixture.assertion_LM_analysis_per_sentence method.
      """
      
      val=self.psycholinguistic_obj.LM_analysis_per_sentence(self.txt)
      assert (np.sum(val[1])==0 and np.sum(val[2])== 12 and np.sum(val[3])==20)

   def assertion_LM_analysis_per_sentence_empty(self):
      """
      This method is used for unit testing StatisticsFixture.assertion_LM_analysis_per_sentence method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.LM_analysis_per_sentence(empty_txt)
      assert ((val[0] is None) and (val[1] is None) and (val[2] is None))

   def assertion_fog_analysis_per_section(self):
      """
      This method is used for unit testing StatisticsFixture.fog_analysis_per_section method.
      """
      
      val=self.psycholinguistic_obj.fog_analysis_per_section(self.txt)
      assert (round(val[0],3)==14.205)

   def assertion_fog_analysis_per_section_empty(self):
      """
      This method is used for unit testing StatisticsFixture.fog_analysis_per_section method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.fog_analysis_per_section(empty_txt)
      assert (np.isnan(val[0]) and np.isnan(val[1]) and np.isnan(val[2]))

   def assertion_fog_analysis_per_sentence(self):
      """
      This method is used for unit testing StatisticsFixture.fog_analysis_per_sentence method.
      """
      
      val=self.psycholinguistic_obj.fog_analysis_per_sentence(self.txt)
      assert (round(0.4*(np.mean(val[2])+100*(np.sum(val[1])/np.sum(val[2]))),3)==14.350)

   def assertion_fog_analysis_per_sentence_empty_text(self):
      """
      This method is used for unit testing StatisticsFixture.fog_analysis empty text method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.fog_analysis_per_sentence(empty_txt)
      assert (val[0] is None and val[1] is None and val[2] is None)

   def assertion_polarity_score_per_section(self):
      """
      This method is used for unit testing StatisticsFixture.polarity_score_per_section method.
      """
      
      val=self.psycholinguistic_obj.polarity_score_per_section(self.txt)
      assert (val[1]==1213 and val[2]== 31 and val[3]==31)

   def assertion_polarity_score_empty_text(self):
      """
      This method is used for unit testing StatisticsFixture.polarity_score_empty_text empty text method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.polarity_score_per_section(empty_txt)
      assert (np.isnan(val[0]))

   def assertion_polarity_score_per_sentence(self):
      """
      This method is used for unit testing StatisticsFixture.polarity_score_per_sentence method.
      """
      
      val=self.psycholinguistic_obj.polarity_score_per_sentence(self.txt)
      assert (np.sum(val[0])==1213 and np.sum(val[1])== 31 and np.sum(val[2])==31)

   def assertion_polarity_score_per_sentence_empty(self):
      """
      This method is used for unit testing StatisticsFixture.polarity_score_per_sentence method.
      """
      empty_txt=""
      val=self.psycholinguistic_obj.polarity_score_per_sentence(empty_txt)
      assert (val[0] is None and val[1] is None and val[2] is None)

result = PsycholinguisticMethodsFixture().execute_tests()
print(result.to_string())
