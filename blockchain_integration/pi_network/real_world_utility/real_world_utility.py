class RealWorldUtility:
    def __init__(self):
        self.industries = []

    def add_industry(self, industry):
        self.industries.append(industry)

    def get_industries(self):
        return self.industries

    def get_industry_by_name(self, name):
        for industry in self.industries:
            if industry.name == name:
                return industry
        return None

    def remove_industry(self, industry):
        self.industries.remove(industry)

    def update_industry(self, industry):
        for i, ind in enumerate(self.industries):
            if ind == industry:
                self.industries[i] = industry
                break

class Industry:
    def __init__(self, name, description):
        self.name = name
        self.description = description

if __name__ == '__main__':
    rwu = RealWorldUtility()
    industry1 = Industry('Finance', 'Financial services and products')
    industry2 = Industry('Healthcare', 'Healthcare services and products')
    rwu.add_industry(industry1)
    rwu.add_industry(industry2)
    print(rwu.get_industries())
    print(rwu.get_industry_by_name('Finance'))
    rwu.remove_industry(industry1)
    print(rwu.get_industries())
    industry1.name = 'Financial Services'
    rwu.update_industry(industry1)
    print(rwu.get_industry_by_name('Financial Services'))
