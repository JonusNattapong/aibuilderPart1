import csv
import uuid

# สร้างข้อความสำหรับหมวดหมู่ crypto
categories = {
    "crypto": [
        "ราคา Bitcoin วันนี้พุ่งสูงขึ้นอีกครั้ง",
        "วิธีใช้งาน Crypto Wallet อย่างปลอดภัย",
        "NFT (Non-Fungible Token) คืออะไร ทำไมถึงมีราคาแพง",
        "Blockchain เทคโนโลยีเบื้องหลังคริปโตเคอร์เรนซี",
        "แนะนำเหรียญ Altcoin ที่น่าจับตามอง",
        "การขุด Bitcoin (Mining) ทำงานอย่างไร",
        "วิธีซื้อขายคริปโตบน Exchange เช่น Binance, Bitkub",
        "ความเสี่ยงในการลงทุนในคริปโตเคอร์เรนซี",
        "DeFi (Decentralized Finance) คืออะไร",
        "การทำ Staking และ Yield Farming ในโลก DeFi",
        "Metaverse โลกเสมือนจริงกับโอกาสทางธุรกิจ",
        "วิธีประเมินมูลค่าเหรียญคริปโตก่อนลงทุน",
        "ข่าวสารล่าสุดเกี่ยวกับกฎระเบียบด้านคริปโตในไทย",
        "การใช้งาน Hardware Wallet เพื่อเก็บเหรียญอย่างปลอดภัย",
        "Smart Contract คืออะไร และทำงานอย่างไรบน Blockchain",
        "แนะนำโปรเจกต์ GameFi ที่น่าสนใจ",
        "วิธีป้องกันตัวเองจากการหลอกลวงในวงการคริปโต",
        "การวิเคราะห์กราฟเทคนิคสำหรับเทรดคริปโต",
        "ผลกระทบของคริปโตเคอร์เรนซีต่อระบบการเงินโลก",
        "DAO (Decentralized Autonomous Organization) คืออะไร",
        "วิธีสร้าง NFT ของตัวเอง",
        "การใช้งาน Decentralized Exchange (DEX) เช่น Uniswap, PancakeSwap",
        "ภาษีคริปโตในประเทศไทย ต้องเสียอย่างไร",
        "อนาคตของ Central Bank Digital Currency (CBDC)",
        "แนะนำ Podcast และแหล่งข้อมูลเกี่ยวกับคริปโต",
        "วิธีเลือก Exchange ที่น่าเชื่อถือ",
        "ความแตกต่างระหว่าง Proof of Work (PoW) และ Proof of Stake (PoS)",
        "การทำ Airdrop และวิธีเข้าร่วม",
        "ผลกระทบของการ Halving ของ Bitcoin",
        "Web3 คืออะไร และแตกต่างจาก Web2 อย่างไร",
        "วิธีใช้งาน DApps (Decentralized Applications)",
        "การลงทุนใน ICO (Initial Coin Offering) มีความเสี่ยงอย่างไร",
        "แนะนำ Influencer ด้านคริปโตที่น่าติดตาม",
        "วิธีตรวจสอบความน่าเชื่อถือของโปรเจกต์คริปโต",
        "การใช้งาน Stablecoin เช่น USDT, USDC, DAI",
        "ผลกระทบด้านสิ่งแวดล้อมของการขุดคริปโต",
        "วิธีโอนเหรียญคริปโตระหว่าง Wallet",
        "การใช้งาน Lending Platform ใน DeFi",
        "ความปลอดภัยของ Smart Contract และความเสี่ยง",
        "อนาคตของ NFT จะเป็นอย่างไรต่อไป",
        "วิธีติดตามข่าวสารและราคาคริปโตแบบเรียลไทม์",
        "การใช้งาน Bridge เพื่อโอนเหรียญข้ามเชน",
        "ผลกระทบของข่าวสารต่อราคาคริปโต",
        "วิธีจัดการพอร์ตการลงทุนคริปโต",
        "การใช้งาน Layer 2 Scaling Solutions เช่น Polygon, Arbitrum",
        "ความเข้าใจผิดเกี่ยวกับ Bitcoin และ Blockchain",
        "วิธีเริ่มต้นลงทุนในคริปโตสำหรับมือใหม่",
        "การใช้งาน Oracle ใน Blockchain",
        "ผลกระทบของสถาบันการเงินต่อตลาดคริปโต",
        "อนาคตของเทคโนโลยี Blockchain นอกเหนือจากคริปโต"
    ]
}

# สร้างรายการข้อมูลพร้อม ID
rows = []
for label, texts in categories.items():
    for text in texts:
        rows.append([str(uuid.uuid4()), text, label])

# บันทึกเป็นไฟล์ CSV
with open('DataOutput/thai_dataset_crypto.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'text', 'label'])
    writer.writerows(rows)

print("Created DataOutput/thai_dataset_crypto.csv")
