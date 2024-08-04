# utility_creation/utility_views.py

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from.models import UtilityUser, UtilityContract, UtilityTransaction
from.forms import UtilityForm
from.utils import create_utility_contract, send_utility_transaction

@login_required
def create_utility_view(request):
    user = request.user
    utility_user = UtilityUser.objects.get(user=user)
    if request.method == 'POST':
        form = UtilityForm(request.POST)
        if form.is_valid():
            mainnet_address = form.cleaned_data['mainnet_address']
            testnet_address = form.cleaned_data['testnet_address']
            utility_user.mainnet_address = mainnet_address
            utility_user.testnet_address = testnet_address
            utility_user.save()
            contract_address, contract_abi = create_utility_contract(utility_user)
            UtilityContract.objects.create(contract_address=contract_address, contract_abi=contract_abi)
            send_utility_transaction(utility_user)
            return redirect('utility_status')
    else:
        form = UtilityForm()
    return render(request, 'create_utility.html', {'form': form})

@login_required
def utility_status_view(request):
    user = request.user
    utility_user = UtilityUser.objects.get(user=user)
    transactions = UtilityTransaction.objects.filter(user=utility_user)
    return render(request, 'utility_status.html', {'transactions': transactions})
